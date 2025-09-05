# Vantage HunyuanFoley 🎵

Modular ComfyUI nodes for HunyuanVideo-Foley that separate model loading, feature encoding (SigLIP2, Synchformer, CLAP), sampling, and VAE decoding. Includes a Foley Sampler with precision control (auto/bf16/fp16/fp32) and optional FP8 quantization (E4M3FN/E5M2) via NVIDIA TransformerEngine when available.

![image](https://github.com/vantagewithai/Vantage-HunyuanFoley/blob/main/src/example_workflows/Hunyuan-Foley.png)

---

## ✨ Highlights

- **Split nodes for lower peak VRAM:**
	 - Text encoder
	 - Vision encoder
	 - Sampler
	 - VAE decode
 - **Precision menu in Sample Loader: auto, bf16, fp16, fp32**
 - **Optional FP8 quantization choices:**
	 - none, fp8_e4m3fn, fp8_e5m2, auto (tries E4M3FN then E5M2)
 - **Per-node “offload to CPU” toggles to free VRAM between stages**
 - **Compatible with the standard ComfyUI IMAGE and AUDIO data types**

## ✅ What this solves

The original monolithic workflow keeps multiple heavy encoders in GPU memory during one pass. Splitting encoders and the denoiser lets each component load, run, then optionally offload, reducing the maximum VRAM spike on 8–12 GB GPUs.

## 📦 What’s included

- Visual Encode (SigLIP2 + Synchformer feature extraction).
- Text Encode (CLAP features, with truncation to cfg length).
- Foley Sampler (runs the native foley denoise process; FP8 autocast aware).
- VAE Loader and VAE Decode (DAC VAE for waveform output).

## 🧰 Requirements
- ComfyUI (current version).
- Python 3.10+ recommended.
- Optional for FP8: NVIDIA Hopper/Blackwell GPU (SM 90+) and TransformerEngine installed.

## ⚙️ Installation

### Method 1: Using ComfyUI Manager (Recommended)

1.  Open ComfyUI Manager.
2.  Click on `Install Custom Nodes`.
3.  Search for `ComfyUI-HunyuanFoley` and click `Install`.
4.  Restart ComfyUI.
5.  Follow the **Download Models** instructions below.

### Method 2: Manual Installation

1.  Navigate to your ComfyUI `custom_nodes` directory.
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/vantagewithai/ComfyUI-HunyuanFoley.git
    ```
3.  Install the required dependencies:
    ```bash
    cd ComfyUI-HunyuanFoley/
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

---

## 📥 Download Models (Crucial Step)

You’ll need to manually download the model files and place them in the required folder structure. This way, the node can run fully offline while giving you complete control.

1.  Navigate to `ComfyUI/models/`.
2.  Create a new folder named `hunyuan_foley`.

3.  **Download the following files and put them inside your 'hunyuan_foley' folder.**

    *   **Hunyuan-Foley Models** from [Tencent/HunyuanVideo-Foley on Hugging Face](https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main):
        *   `hunyuanvideo_foley.pth`
        *   `synchformer_state_dict.pth`
        *   `vae_128d_48k.pth`

    *   **SigLIP Vision Model** from [google/siglip2-base-patch16-512 on Hugging Face](https://huggingface.co/google/siglip2-base-patch16-512/tree/main):
        *   Create a new folder named `siglip2`.
        *   Download `model.safetensors`, `config.json` and `preprocessor_config.json` place them inside the `siglip2` folder.

    *   **CLAP Text Model** from [laion/larger_clap_general on Hugging Face](https://huggingface.co/laion/larger_clap_general/tree/refs%2Fpr%2F2):
        *   Create a new folder named `clap`.
        *   Download `model.safetensors`, `config.json`, `merges.txt` and `vocab.json` and place them inside the `clap` folder.

**Your final folder structure should look exactly like this:**

```
ComfyUI/
└── models/
    └── hunyuan_foley/ 
        ├── hunyuanvideo_foley.pth
        ├── synchformer_state_dict.pth
        ├── vae_128d_48k.pth
        │
        ├── siglip2/          <-- SigLIP2
        │   ├── model.safetensors
        │   ├── config.json
        │   └── preprocessor_config.json
        │
        └── clap/             <-- CLAP
            ├── model.safetensors
            ├── config.json
            ├── merges.txt
            └── vocab.json
```
[![Watch the video](https://github.com/vantagewithai/Vantage-HunyuanFoley/blob/main/src/example_workflows/video.png)](https://youtu.be/9HsC445ChhI)

## 🚀 Usage & Nodes

The workflow is designed to be modular and familiar to ComfyUI users.

These custom nodes provide an end-to-end pipeline to generate Foley audio conditioned on video frames and text prompts, with optional FP8 acceleration on supported NVIDIA GPUs. Nodes include visual and text encoders, a sampler/denoiser, and a DAC VAE decoder, organized for low-VRAM usage with optional offload steps

#### 1. `Hunyuan-Foley Visual Encode`
-   **Class:** HunyuanVisualEncode; Category: VantageWithAI/HunyuanFoley.
-   **Purpose:** Extract video-conditioned features:
    -   SigLIP2 pooled embeddings sampled at FPS_VISUAL["siglip2"].
    -   Synchformer audio-visual synchronization embeddings sampled at FPS_VISUAL["synchformer"].
-   **Inputs:**
    -   **video_frames:** IMAGE (batched frames normalized to 0–1 as in ComfyUI).
    -   **fps:** FLOAT (default 24.0).
    -   **to_gpu:** BOOLEAN (move encoders to GPU if available)
    -   **move_back_to_cpu:** BOOLEAN (offload encoders after use).
    -   **seed:** INT (for reproducible processing order and any RNG use).
-   **Outputs:**
    -   **siglip2:** SIGLIP2_FEAT.
    -   **syncformer:** SYNC_FEAT.
    -   **audio_len:** AUDIO_LEN (seconds derived from num frames and fps)
-   **Notes:** Requires siglip2/ and synchformer_state_dict.pth to exist under models/hunyuan_foley

#### 2. `Hunyuan-Foley Text Encode`
-   **Class:** HunyuanTextEncode; Category: VantageWithAI/HunyuanFoley.
-   **Purpose:** Encode CLAP text features for positive and negative prompts and truncate to configured max text length.
-   **Inputs:**
    -   **prompt:** STRING (positive).
    -   **negative_prompt:** STRING.
    -   **to_gpu:** BOOLEAN (move encoders to GPU if available)
    -   **move_back_to_cpu:** BOOLEAN (offload encoders after use).
-   **Outputs:**
    -   **positive:** TEXT_FEAT.
    -   **negative:** UNCOND_TEXT_FEAT.
-   **Notes:** Requires clap/ Hugging Face directory with tokenizer and ClapTextModelWithProjection.

#### 3. `Hunyuan-Foley Denoiser`
-   **Class:** HunyuanFoleyDenoiser; Category: VantageWithAI/HunyuanFoley.
-   **Purpose:** Construct and run the HunyuanVideoFoley denoiser using the provided visual and text features, producing latents for the DAC VAE.
-   **Inputs:**
    -   **siglip2:** SIGLIP2_FEAT from Visual Encode.
    -   **syncformer:** SYNC_FEAT from Visual Encode.
    -   **positive:** TEXT_FEAT from Text Encode.
    -   **negative:** UNCOND_TEXT_FEAT from Text Encode.
    -   **audio_len:** AUDIO_LEN (seconds).
    -   **guidance_scale:** FLOAT (default 4.5).
    -   **steps:** INT (default 50).
    -   **seed:** INT for reproducibility.
    -   **precision:** auto/bf16/fp16/fp32 for non-FP8 paths.
    -   **quantization:** none/fp8_e4m3fn/fp8_e5m2/auto; requires NVIDIA TransformerEngine and SM 90+ for FP8 to take effect.
    -   **to_gpu:** BOOLEAN (move encoders to GPU if available)
    -   **move_back_to_cpu:** BOOLEAN (offload encoders after use).
-   **Outputs:*
    -   **samples:** LATENT, with {"samples": latents, "audio_len_in_s": float(audio_len)}.
-   **Notes:**
    -   Loads hunyuanvideo_foley.pth and the YAML config under src/hunyuanvideo_foley/configs; strict=False state_dict load tolerates minor key differences.
    -   FP8 path replaces nn.Linear with TransformerEngine te.Linear and wraps denoising in te.fp8_autocast if supported; otherwise runs standard mixed/bfloat16/float32 as selected.

#### 4. `Hunyuan-Foley VAE Decode`
-   **Class:** HunyuanFoleyVAEDecode; Category: VantageWithAI/HunyuanFoley.
-   **Purpose:** Decode DAC VAE latents to waveform audio at the DAC’s sample_rate and trim to requested duration.
-   **Inputs:**
    -   **samples:** LATENT from the sampler (contains latents and audio_len_in_s).
    -   **to_gpu:** BOOLEAN (move encoders to GPU if available)
    -   **move_back_to_cpu:** BOOLEAN (offload encoders after use).
-   **Outputs:**
    -   **audio:** AUDIO dict with keys waveform (Tensor [B,T]) and sample_rate (int).
-   **Notes:** Loads vae_128d_48k.pth from models/hunyuan_foley.

### 💡 Typical workflow
Load video frames with VHS_LoadVideo, extract fps with VHS_VideoInfo, encode visual and text features, sample Foley latents, decode to audio, and mux video+audio with VHS_VideoCombine.

**Node bindings**
1. Video loading and info
    -   VHS_LoadVideo → outputs:
        -   IMAGE → connect to HunyuanVisualEncode.video_frames and to VHS_VideoCombine.images.
        -   VHS_VIDEOINFO → connect to VHS_VideoInfo.video_info.
    -   VHS_VideoInfo → outputs:
        -   loaded_fps🟦 (FLOAT) → connect to HunyuanVisualEncode.fps.
        -   loaded_fps🟦 (FLOAT) → also connect to VHS_VideoCombine.frame_rate.
Connections:
    -   VHS_LoadVideo.IMAGE → HunyuanVisualEncode.video_frames.
    -   VHS_LoadVideo.IMAGE → VHS_VideoCombine.images.
    -   VHS_LoadVideo.VHS_VIDEOINFO → VHS_VideoInfo.video_info.
    -   VHS_VideoInfo.loaded_fps🟦 → HunyuanVisualEncode.fps.
    -   VHS_VideoInfo.loaded_fps🟦 → VHS_VideoCombine.frame_rate.
Tips:
    -   Keep VHS_LoadVideo “select_every_nth” aligned with target fps/latency needs; the graph uses loaded fps for both encoding cadence and mux fps.

2. Visual feature encoding
    -   HunyuanVisualEncode → inputs:
        -   video_frames from VHS_LoadVideo.IMAGE (already wired above).
        -   fps from VHS_VideoInfo.loaded_fps🟦 (already wired above).
    -   HunyuanVisualEncode → outputs:
        -   siglip2 → connect to HunyuanFoleySampler.siglip2.
        -   syncformer → connect to HunyuanFoleySampler.syncformer.
        -   audio_len → connect to HunyuanFoleySampler.audio_len.
Connections:
    -   HunyuanVisualEncode.siglip2 → HunyuanFoleySampler.siglip2.
    -   HunyuanVisualEncode.syncformer → HunyuanFoleySampler.syncformer.
    -   HunyuanVisualEncode.audio_len → HunyuanFoleySampler.audio_len.
Notes:
    -   The node samples frames internally for SigLIP2/Synchformer based on fps and FPS_VISUAL configuration; ensure SigLIP2 and Synchformer assets exist under models/hunyuan_foley.

3. Text feature encoding
    -   HunyuanTextEncode → outputs:
        -   positive → connect to HunyuanFoleySampler.positive.
        -   negative → connect to HunyuanFoleySampler.negative.
Connections:
    -   HunyuanTextEncode.positive → HunyuanFoleySampler.positive.
    -   HunyuanTextEncode.negative → HunyuanFoleySampler.negative.
Tips:
    -   Adjust prompt/negative_prompt to the scene; keep within max text length enforced by the config.

4. Foley sampling/denoising
    -   HunyuanFoleySampler → inputs:
        -   siglip2 from HunyuanVisualEncode.siglip2.
        -   syncformer from HunyuanVisualEncode.syncformer.
        -   positive from HunyuanTextEncode.positive.
        -   negative from HunyuanTextEncode.negative.
        -   audio_len from HunyuanVisualEncode.audio_len.
    -   HunyuanFoleySampler → outputs:
        -   samples (LATENT) → connect to HunyuanFoleyVAEDecode.samples.
Connection:
    -   HunyuanFoleySampler.samples → HunyuanFoleyVAEDecode.samples.
Parameters:
    -   guidance_scale = 4.5 (default), steps = 50, seed randomized or fixed as needed; precision = auto; quantization = none by default. Adjust for quality vs. speed/VRAM.

5. VAE decode to audio
    -   HunyuanFoleyVAEDecode → input:
        -   samples from HunyuanFoleySampler.samples.
    -   HunyuanFoleyVAEDecode → outputs:
        -   audio (AUDIO) → connect to VHS_VideoCombine.audio.
Connection:
    -   HunyuanFoleyVAEDecode.audio → VHS_VideoCombine.audio.
Notes:
    -   The DAC VAE trims waveform to audio_len_in_s embedded in samples; ensure vae_128d_48k.pth is in models/hunyuan_foley or the global models directory.

6. Mux video and audio, save
    -   VHS_VideoCombine → inputs:
        -   images from VHS_LoadVideo.IMAGE (already wired).
        -   audio from HunyuanFoleyVAEDecode.audio.
        -   frame_rate from VHS_VideoInfo.loaded_fps🟦 (already wired).
    -   VHS_VideoCombine → outputs:
        -   Filenames (VHS_FILENAMES) → connect to VHS_PruneOutputs.filenames if desired to prune intermediates.
Connections:
    -   VHS_VideoCombine.Filenames → VHS_PruneOutputs.filenames.
Tips:
    -   Set format, pix_fmt, and crf per target; the example uses video/h265-mp4, yuv420p10le, crf 19, frame_rate 16. Verify external players support your choices.

Execution order
    -   Nodes naturally resolve by dependencies: LoadVideo → VideoInfo → VisualEncode/TextEncode → Sampler → VAEDecode → VideoCombine → PruneOutputs. Trigger execution from VideoCombine or any downstream node to build the complete media output.

Practical tips
    -   **VRAM:** Enable move_back_to_cpu on encode and decode nodes to reduce peaks on 8–12 GB GPUs; lower steps if running out of memory.
    -   **Reproducibility:** Fix seed in HunyuanFoleySampler; align fps between VisualEncode and VideoCombine to maintain sync.
    -   **Debugging:** Watch ComfyUI console logs for missing files or dtype/device messages; the nodes log clear errors on missing assets.

Wire these nodes in the ComfyUI graph; nodes appear under VantageWithAI/HunyuanFoley in the right-click menu.

## 🙏 Acknowledgements

-   **Tencent Hunyuan:** For creating and open-sourcing the original [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) model.
-   **Google Research:** For the SigLIP model.
-   **LAION:** For the CLAP model.
-   **Descript:** For the [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) (DAC VAE).

-   **v-iashin:** For the [Synchformer](https://github.com/v-iashin/Synchformer) model.











