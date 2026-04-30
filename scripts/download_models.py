from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


FILES = [
    {
        "label": "FLUX.2-Klein 4B base diffusion model",
        "repo": "black-forest-labs/FLUX.2-klein-base-4b-fp8",
        "filename": "flux-2-klein-base-4b-fp8.safetensors",
        "folder": "diffusion_models",
        "target": "flux-2-klein/flux-2-klein-4b-fp8.safetensors",
    },
    {
        "label": "RUM FLUX.2-Klein preview checkpoint",
        "repo": "rimochan/RUM-FLUX.2-klein-4B-preview",
        "filename": "model-checkpoint-608000.safetensors",
        "folder": "diffusion_models",
        "target": "rum-flux2-klein-4b-preview.safetensors",
    },
    {
        "label": "FLUX.2-Klein Qwen3 4B text encoder",
        "repo": "Comfy-Org/z_image_turbo",
        "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
        "folder": "text_encoders",
        "target": "qwen_3_4b.safetensors",
    },
    {
        "label": "FLUX.2 VAE",
        "repo": "Comfy-Org/flux2-dev",
        "filename": "split_files/vae/flux2-vae.safetensors",
        "folder": "vae",
        "target": "flux2-vae.safetensors",
    },
]

OPTIONAL_SDXL_CLIP_FILES = [
    {
        "label": "SDXL/SD3 CLIP-L text encoder",
        "repo": "Comfy-Org/stable-diffusion-3.5-fp8",
        "filename": "text_encoders/clip_l.safetensors",
        "folder": "text_encoders",
        "target": "clip_l.safetensors",
    },
    {
        "label": "SDXL/SD3 CLIP-G text encoder",
        "repo": "Comfy-Org/stable-diffusion-3.5-fp8",
        "filename": "text_encoders/clip_g.safetensors",
        "folder": "text_encoders",
        "target": "clip_g.safetensors",
    },
]


def parse_args() -> argparse.Namespace:
    default_comfy_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Download native ComfyUI model files for ComfyUI-RUM.")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=default_comfy_root,
        help="ComfyUI root directory. Default: inferred from custom_nodes/ComfyUI-RUM/scripts.",
    )
    parser.add_argument(
        "--include-sdxl-clip",
        action="store_true",
        help="Also try downloading clip_l.safetensors and clip_g.safetensors for DualCLIPLoader type=sdxl.",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    return parser.parse_args()


def download_one(item: dict[str, str], models_dir: Path, token: str | None) -> Path:
    target_dir = models_dir / item["folder"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / item["target"]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        print(f"SKIP {item['label']}: {target_path}")
        return target_path

    print(f"Downloading {item['label']} -> {target_path}")
    cached_path = hf_hub_download(repo_id=item["repo"], filename=item["filename"], token=token)
    shutil.copy2(cached_path, target_path)
    return target_path


def main() -> None:
    args = parse_args()
    comfy_root = args.comfy_root.expanduser().resolve()
    models_dir = comfy_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    selected = FILES + (OPTIONAL_SDXL_CLIP_FILES if args.include_sdxl_clip else [])
    written = [download_one(item, models_dir, args.token) for item in selected]

    print("\nDone. Restart ComfyUI, then select these files in normal ComfyUI loader nodes:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
