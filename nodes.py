# MIT License
# Copyright (c) 2025 Vantage with AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Vantage with AI HunyuanVideoFoley ComfyUI nodes with optional FP8 quantization.

Author: Vantage with AI
License: MIT
Summary:
    - Splits the original monolithic Hunyuan Foley node into modular loaders and encoders.
    - Adds a Foley model loader with precision (auto/bf16/fp16/fp32) and FP8 quantization options
      (E4M3FN/E5M2/auto) using NVIDIA TransformerEngine when available.
    - Supports low-VRAM usage via per-node GPU offload toggles and empty CUDA cache calls.
    - Keeps the original inference flow: visual/text encoders -> denoiser -> DAC VAE decode.

Folder layout under ComfyUI/models/hunyuan_foley:
    - hunyuanvideo_foley.pth
    - clap/ (HF CLAP)
    - siglip2/ (HF SigLIP2)
    - synchformer_state_dict.pth
    - Optionally VAE: vae_128d_48k.pth under the same tree or in the global 'vae' directory.

Notes:
    - FP8 requires Hopper/Blackwell-class GPUs (SM 90+) and 'transformer-engine' installed.
    - If FP8 is unavailable, the loader falls back to the selected base precision automatically.
"""
__author__ = "Vantage with AI"
__license__ = "MIT"
__version__ = "0.1.0"

import os
import gc
import logging
import random
import numpy as np
import torch
from PIL import Image
import folder_paths

from .src.hunyuanvideo_foley.utils.config_utils import load_yaml, AttributeDict
from .src.hunyuanvideo_foley.utils.feature_utils import encode_video_with_sync, encode_text_feat
from .src.hunyuanvideo_foley.utils.model_utils import denoise_process
from .src.hunyuanvideo_foley.constants import FPS_VISUAL
from .src.hunyuanvideo_foley.models.dac_vae.model.dac import DAC

from transformers import (
    AutoTokenizer,
    ClapTextModelWithProjection,
    SiglipImageProcessor,
    SiglipVisionModel,
)

logging.basicConfig(level=logging.INFO, format='vantage-with-ai/HunyuanFoley (%(levelname)s): %(message)s') # Configure simple console logging prefix for easier debugging in ComfyUI logs. [info/warn/error]

# Seed utilities for reproducibility across CUDA/CPU; ComfyUI often reuses graph runs. [helpers]
def set_manual_seed(seed: int):
    seed = int(seed)
    numpy_seed = seed % (2**32)
    random.seed(numpy_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _models_root():
    return os.path.join(folder_paths.models_dir, "hunyuan_foley")

def _ensure_exists(p, msg):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{msg}: {p}")

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Precision resolver:
# - "auto": prefer bf16 on supported CUDA, else fp16, else fp32.
# - explicit: bf16/fp16/fp32 map to torch dtypes for non-FP8 layers. 
def _dtype(precision: str):
    if precision == "auto":
        # prefer bf16 if available (Ampere+), else fp16 on CUDA, else fp32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[precision]

# -------------------------
# Encoders 
# Optional offload to reduce peak VRAM across stages; recommended on 8–12 GB GPUs.
# -------------------------
class HunyuanVisualEncode:
    CATEGORY = "VantageWithAI/HunyuanFoley"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "video_frames": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            "to_gpu": ("BOOLEAN", {"default": True}),
            "move_back_to_cpu": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        }}

    RETURN_TYPES = ("SIGLIP2_FEAT", "SYNC_FEAT", "AUDIO_LEN")
    RETURN_NAMES = ("siglip2", "syncformer", "audio_len")
    FUNCTION = "encode"

    def encode(self, video_frames, fps, to_gpu=True, move_back_to_cpu=True, seed=0):
        from torchvision.transforms import v2
        from .src.hunyuanvideo_foley.models.synchformer import Synchformer
        
        set_manual_seed(seed)
        device = _device()

        preprocess = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        model = Synchformer().eval()
        state_pth = os.path.join(_models_root(), "synchformer_state_dict.pth")
        _ensure_exists(state_pth, "Synchformer state dict not found")
        model.load_state_dict(torch.load(state_pth, map_location="cpu"))

        if to_gpu and torch.cuda.is_available():
            model = model.to("cuda")
        
        siglip_path = os.path.join(_models_root(), "siglip2")
        _ensure_exists(siglip_path, "SigLIP2 folder not found")
        preprocess_siglip = SiglipImageProcessor.from_pretrained(siglip_path, local_files_only=True)
        model_siglip = SiglipVisionModel.from_pretrained(siglip_path, local_files_only=True, low_cpu_mem_usage=True).eval()
        if to_gpu and torch.cuda.is_available():
            model_siglip = model_siglip.to("cuda")
        
        frames_np = (video_frames.cpu().numpy() * 255).astype(np.uint8)
        all_frames = [f for f in frames_np]
        num_frames = len(all_frames)
        audio_len_in_s = float(num_frames) / float(fps)

        siglip_fps = FPS_VISUAL["siglip2"]
        siglip_indices = np.linspace(0, num_frames - 1, max(1, int(audio_len_in_s * siglip_fps))).astype(int)
        frames_siglip = [all_frames[i] for i in siglip_indices]
        images_siglip = preprocess_siglip(
            images=[Image.fromarray(f).convert('RGB') for f in frames_siglip],
            return_tensors="pt"
        )

        if device == "cuda":
            images_siglip = {k: v.to(device) for k, v in images_siglip.items()}
            siglip2_model = model_siglip.to(device)
        else:
            siglip2_model = model_siglip

        with torch.no_grad():
            siglip_output = siglip2_model(**images_siglip)
            siglip_feat = siglip_output.pooler_output.unsqueeze(0).contiguous()

        sync_fps = FPS_VISUAL["synchformer"]
        sync_indices = np.linspace(0, num_frames - 1, max(1, int(audio_len_in_s * sync_fps))).astype(int)
        frames_sync_np = np.array([all_frames[i] for i in sync_indices])
        images_sync = torch.from_numpy(frames_sync_np).permute(0, 3, 1, 2)
        sync_frames = preprocess(images_sync).unsqueeze(0)

        model_dict_with_device = {
            "syncformer_model": model.to(device) if device == "cuda" else model,
            "device": device
        }

        with torch.no_grad():
            sync_feat = encode_video_with_sync(sync_frames.to(device), AttributeDict(model_dict_with_device))

        if move_back_to_cpu:
            try:
                model_siglip.to("cpu")
                model.to("cpu")
            except Exception:
                pass
            empty_cuda_cache()

        return (siglip_feat.cpu(), sync_feat.cpu(), audio_len_in_s,)

class HunyuanTextEncode:
    CATEGORY = "VantageWithAI/HunyuanFoley"

    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "A person walks on frozen ice"}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "noisy, harsh"}),
            "to_gpu": ("BOOLEAN", {"default": True}),
            "move_back_to_cpu": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("TEXT_FEAT", "UNCOND_TEXT_FEAT")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"

    def encode(self, prompt, negative_prompt, to_gpu=True, move_back_to_cpu=True):
        clap_path = os.path.join(_models_root(), "clap")
        _ensure_exists(clap_path, "CLAP folder not found")
        tokenizer = AutoTokenizer.from_pretrained(clap_path, local_files_only=True)
        model = ClapTextModelWithProjection.from_pretrained(clap_path, local_files_only=True, low_cpu_mem_usage=True).eval()
        if to_gpu and torch.cuda.is_available():
            model = model.to("cuda")
        
        device = _device()
        model_dict = {
            "clap_tokenizer": tokenizer,
            "clap_model": model.to(device) if device == "cuda" else clap["clap_model"],
            "device": device
        }
        prompts = [negative_prompt, prompt]
        
        module_dir = os.path.dirname(__file__)
        config_path = os.path.join(module_dir, "src/hunyuanvideo_foley/configs", "hunyuanvideo-foley-xxl.yaml")
        _ensure_exists(config_path, "Config not found")
        cfg = load_yaml(config_path)
        
        with torch.no_grad():
            text_feat_res, _ = encode_text_feat(prompts, AttributeDict(model_dict))
        text_feat, uncond_text_feat = text_feat_res[1:], text_feat_res[:1]

        max_len = cfg.model_config.model_kwargs.text_length
        if text_feat.shape[1] > max_len:
            text_feat = text_feat[:, :max_len]
            uncond_text_feat = uncond_text_feat[:, :max_len]

        if move_back_to_cpu:
            try:
                clap["clap_model"].to("cpu")
            except Exception:
                pass
            empty_cuda_cache()

        return (text_feat.cpu(), uncond_text_feat.cpu(),)
                
# -------------------------
# Sampler
# -------------------------
class HunyuanFoleyDenoiser:
    CATEGORY = "VantageWithAI/HunyuanFoley"

    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "siglip2": ("SIGLIP2_FEAT",),
            "syncformer": ("SYNC_FEAT",),
            "positive": ("TEXT_FEAT",),
            "negative": ("UNCOND_TEXT_FEAT",),
            "audio_len": ("AUDIO_LEN",),
            "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            "steps": ("INT", {"default": 50, "min": 10, "max": 200, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
            "quantization": (["none", "fp8_e4m3fn", "fp8_e5m2", "auto"], {"default": "none"}),
            "to_gpu": ("BOOLEAN", {"default": True}),
            "move_back_to_cpu": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "denoise"
    
    # FP8 helpers:
#   _swap_linear_to_te: Recursively replaces nn.Linear with transformer_engine.te.Linear.
#   _build_fp8_recipe: Configures E4M3FN or E5M2 scaling via DelayedScaling.
# Requirements:
#   - NVIDIA TransformerEngine installed.
#   - GPU with SM 90+ (Hopper/Blackwell) for FP8 kernels; otherwise, falls back gracefully.
    def _swap_linear_to_te(self, model):
        # Replace torch.nn.Linear with TE te.Linear in-place
        try:
            import transformer_engine.pytorch as te  # noqa: F401
        except Exception as e:
            logging.warning(f"TransformerEngine import failed, cannot swap to FP8 Linear: {e}")
            return False

        import torch.nn as nn
        import transformer_engine.pytorch as te

        replaced = 0
        for name, module in list(model.named_modules()):
            # walk children only; avoid replacing the top module itself incorrectly
            pass
        # Recursive replace
        def convert(module):
            import torch.nn as nn
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    ln = te.Linear(child.in_features, child.out_features, bias=(child.bias is not None))
                    with torch.no_grad():
                        ln.weight.copy_(child.weight)
                        if child.bias is not None:
                            ln.bias.copy_(child.bias)
                    setattr(module, name, ln)
                    nonlocal replaced
                    replaced += 1
                else:
                    convert(child)
        convert(model)
        logging.info(f"TransformerEngine swap: replaced {replaced} Linear layers with te.Linear")
        if replaced == 0:
            logging.warning("No nn.Linear layers were replaced; FP8 may have no effect.")
        # TE lin kernels require dims divisible by 16; warn but continue
        logging.info("Note: TE FP8 Linear prefers dims divisible by 16; incompatible shapes may fall back or error.")
        return replaced > 0

    def _build_fp8_recipe(self, fmt):
        # Build TE recipe for chosen FP8 format
        try:
            from transformer_engine.common.recipe import Format, DelayedScaling
        except Exception as e:
            logging.warning(f"TransformerEngine recipe import failed: {e}")
            return None

        fmt_map = {
            "fp8_e4m3fn": "E4M3",
            "fp8_e5m2": "E5M2",
        }
        te_fmt = getattr(__import__("transformer_engine.common.recipe", fromlist=["Format"]).Format, fmt_map.get(fmt, "E4M3"))
        # Using delayed scaling with short amax history for inference
        return DelayedScaling(fp8_format=te_fmt, amax_history_len=16, amax_compute_algo="max")
    
    def denoise(self, siglip2, syncformer, positive, negative,
                audio_len, guidance_scale, steps, seed, precision="auto", quantization="none", to_gpu=True, move_back_to_cpu=True):
        from .src.hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
        
        precision_txt = precision or "auto"
        dtype = _dtype(precision_txt)
        
        module_dir = os.path.dirname(__file__)
        config_path = os.path.join(module_dir, "src/hunyuanvideo_foley/configs", "hunyuanvideo-foley-xxl.yaml")
        _ensure_exists(config_path, "Config not found")
        cfg = load_yaml(config_path)
        
        model = HunyuanVideoFoley(cfg, dtype=dtype).eval()
        foley_pth = os.path.join(_models_root(), "hunyuanvideo_foley.pth")
        _ensure_exists(foley_pth, "HunyuanVideo Foley checkpoint not found")
        state = torch.load(foley_pth, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        
        # Apply FP8 if requested and possible
        fp8_enabled = False
        fp8_recipe = None
        selected_fmt = None

        if quantization in ("fp8_e4m3fn", "fp8_e5m2", "auto"):
            if not torch.cuda.is_available():
                logging.warning("FP8 requested but CUDA not available; continuing without FP8.")
            else:
                try:
                    import transformer_engine.pytorch as te  # ensure import
                    cc_major = torch.cuda.get_device_capability()
                    if cc_major < 9:
                        logging.warning("FP8 requires Hopper/Blackwell class GPU (SM 90+). Falling back without FP8.")
                    else:
                        choice_order = [quantization] if quantization != "auto" else ["fp8_e4m3fn", "fp8_e5m2"]
                        for choice in choice_order:
                            fmt_recipe = self._build_fp8_recipe(choice)
                            if fmt_recipe is None:
                                continue
                            swapped = self._swap_linear_to_te(model)
                            if swapped:
                                fp8_enabled = True
                                fp8_recipe = fmt_recipe
                                selected_fmt = choice
                                break
                        if fp8_enabled:
                            model._use_te_fp8 = True
                            model._fp8_recipe = fp8_recipe
                            model._fp8_format = selected_fmt
                            logging.info(f"FP8 enabled with format: {selected_fmt}")
                        else:
                            logging.warning("FP8 setup attempted but no layers were swapped; running without FP8.")
                except Exception as e:
                    logging.warning(f"FP8 setup failed: {e}. Running without FP8.")

        if to_gpu and torch.cuda.is_available():
            # Keep base compute dtype for non-TE parts
            model = model.to("cuda", dtype=dtype)
        
        set_manual_seed(seed)
        device = _device()
        
        module_dir = os.path.dirname(__file__)
        config_path = os.path.join(module_dir, "src/hunyuanvideo_foley/configs", "hunyuanvideo-foley-xxl.yaml")
        _ensure_exists(config_path, "Config not found")
        cfg = load_yaml(config_path)
        
        fm = model.to(device, dtype=dtype) if device == "cuda" else model

        visual_feats = AttributeDict({
            "siglip2_feat": siglip2.to(device),
            "syncformer_feat": syncformer.to(device),
        })
        text_feats = AttributeDict({
            "text_feat": positive.to(device),
            "uncond_text_feat": negative.to(device),
        })
        model_wrap = AttributeDict({
            "foley_model": fm,
            "device": device,
        })

        use_te_fp8 = getattr(fm, "_use_te_fp8", False)
        if use_te_fp8:
            try:
                import transformer_engine.pytorch as te
                fp8_recipe = getattr(fm, "_fp8_recipe", None)
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    with torch.no_grad():
                        latents = denoise_process(visual_feats, text_feats, float(audio_len), model_wrap, cfg, float(guidance_scale), int(steps))
            except Exception as e:
                logging.warning(f"FP8 autocast failed, running without FP8: {e}")
                with torch.no_grad():
                    latents = denoise_process(visual_feats, text_feats, float(audio_len), model_wrap, cfg, float(guidance_scale), int(steps))
        else:
            with torch.no_grad():
                latents = denoise_process(visual_feats, text_feats, float(audio_len), model_wrap, cfg, float(guidance_scale), int(steps))

        if move_back_to_cpu:
            try:
                model.to("cpu")
            except Exception:
                pass
            empty_cuda_cache()

        return ({"samples": latents.cpu(), "audio_len_in_s": float(audio_len)},)

class HunyuanFoleyVAEDecode:
    CATEGORY = "VantageWithAI/HunyuanFoley"


    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "to_gpu": ("BOOLEAN", {"default": True}),
            "move_back_to_cpu": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"

    def decode(self, samples, to_gpu=True, move_back_to_cpu=True):
        vae_path = os.path.join(_models_root(), "vae_128d_48k.pth")
        _ensure_exists(vae_path, "VAE file not found")
       
        vae = DAC.load(vae_path).eval()
        if to_gpu and torch.cuda.is_available():
            vae = vae.to("cuda")
        
        latents = samples["samples"]
        audio_len_in_s = samples["audio_len_in_s"]
        device = _device()

        vae_on_device = vae.to(device)
        try:
            with torch.no_grad():
                audio_tensor = vae_on_device.decode(latents.to(device)).float().cpu()
                sample_rate = vae.sample_rate
                audio_tensor = audio_tensor[:, :int(float(audio_len_in_s) * sample_rate)]
                audio_out = {"waveform": audio_tensor, "sample_rate": int(sample_rate)}
        finally:
            if move_back_to_cpu:
                vae_on_device.to("cpu")
                empty_cuda_cache()

        return (audio_out,)

