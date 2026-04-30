from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


FLUX_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
RUM_REPO = "rimochan/RUM-FLUX.2-klein-4B-preview"
RUM_FILE = "model-checkpoint-608000.safetensors"
SDXL_TEXT_REPO = "Ine007/waiIllustriousSDXL_v160"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model files required by ComfyUI-RUM.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Directory where model folders will be written. Default: ./models inside this custom node.",
    )
    parser.add_argument(
        "--write-local-config",
        action="store_true",
        help="Write local_config.json with the downloaded paths.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for rate limits or gated models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    flux_dir = models_dir / "FLUX.2-klein-base-4B"
    rum_dir = models_dir / "RUM-FLUX.2-klein-4B-preview"
    sdxl_dir = models_dir / "waiIllustriousSDXL_v160_text"

    print(f"Downloading FLUX.2-klein base to: {flux_dir}")
    snapshot_download(
        repo_id=FLUX_REPO,
        local_dir=flux_dir,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ],
        token=args.token,
    )

    print(f"Downloading RUM checkpoint to: {rum_dir}")
    hf_hub_download(
        repo_id=RUM_REPO,
        filename=RUM_FILE,
        local_dir=rum_dir,
        token=args.token,
    )

    print(f"Downloading SDXL text encoders to: {sdxl_dir}")
    snapshot_download(
        repo_id=SDXL_TEXT_REPO,
        local_dir=sdxl_dir,
        allow_patterns=[
            "model_index.json",
            "text_encoder/*",
            "text_encoder_2/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ],
        token=args.token,
    )

    config = {
        "base_model_path": str(flux_dir),
        "rum_checkpoint_path": str(rum_dir / RUM_FILE),
        "sdxl_model_path": str(sdxl_dir),
    }

    if args.write_local_config:
        config_path = Path(__file__).resolve().parents[1] / "local_config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote local config: {config_path}")

    print("\nUse these paths in the RUM FLUX.2-klein Loader node:")
    for key, value in config.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
