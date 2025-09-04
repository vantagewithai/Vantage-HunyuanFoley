from .nodes import HunyuanVisualEncode, HunyuanTextEncode, HunyuanFoleyDenoiser, HunyuanFoleyVAEDecode

NODE_CLASS_MAPPINGS = {
    "HunyuanVisualEncode": HunyuanVisualEncode,
    "HunyuanTextEncode": HunyuanTextEncode,
    "HunyuanFoleyDenoiser": HunyuanFoleyDenoiser,
    "HunyuanFoleyVAEDecode": HunyuanFoleyVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVisualEncode": "HunyuanFoley Visual Encode",
    "HunyuanTextEncode": "HunyuanFoley Text Encode",
    "HunyuanFoleyDenoiser": "HunyuanFoley Denoiser",
    "HunyuanFoleyVAEDecode": "HunyuanFoley VAE Decode",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
