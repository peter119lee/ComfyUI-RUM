from __future__ import annotations

from pathlib import Path


NODE_DIR = Path(__file__).resolve().parent


def _diffusion_model_list():
    try:
        import folder_paths

        names = folder_paths.get_filename_list("diffusion_models")
        return names if names else ["put_rum_checkpoint_in_models_diffusion_models.safetensors"]
    except Exception:
        return ["put_rum_checkpoint_in_models_diffusion_models.safetensors"]


class RUMFlux2ApplyModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rum_checkpoint_name": (_diffusion_model_list(),),
                "base_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "strict": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "rum_checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：填写绝对路径时会覆盖 rum_checkpoint_name。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply_patch"
    CATEGORY = "RUM/native"

    def apply_patch(self, model, rum_checkpoint_name: str, base_text_tokens: int, strict: bool, rum_checkpoint_path: str = ""):
        import folder_paths

        from .rum_native import apply_rum_model_patch

        checkpoint = rum_checkpoint_path.strip()
        if not checkpoint:
            checkpoint = folder_paths.get_full_path_or_raise("diffusion_models", rum_checkpoint_name)

        patched, count, resolved = apply_rum_model_patch(
            model,
            checkpoint,
            base_text_tokens=base_text_tokens,
            strict=strict,
        )
        return (patched, f"RUM native patch 已套用：{count} 个权重；checkpoint={resolved}")


class RUMFlux2CombineConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux2_conditioning": ("CONDITIONING",),
                "sdxl_conditioning": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "base_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "sdxl_clip_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "RUM/native"

    def combine(
        self,
        flux2_conditioning,
        sdxl_conditioning,
        guidance: float,
        base_text_tokens: int,
        extra_text_tokens: int,
        sdxl_clip_width: int,
    ):
        from .rum_native import combine_rum_conditioning

        return (
            combine_rum_conditioning(
                flux2_conditioning,
                sdxl_conditioning,
                guidance=guidance,
                base_text_tokens=base_text_tokens,
                extra_text_tokens=extra_text_tokens,
                sdxl_clip_width=sdxl_clip_width,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "RUMFlux2ApplyModelPatch": RUMFlux2ApplyModelPatch,
    "RUMFlux2CombineConditioning": RUMFlux2CombineConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RUMFlux2ApplyModelPatch": "RUM FLUX.2 Apply Model Patch",
    "RUMFlux2CombineConditioning": "RUM FLUX.2 Combine Conditioning",
}
