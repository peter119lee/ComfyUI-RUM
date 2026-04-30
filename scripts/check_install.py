from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_node_module():
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_RUM_check",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create module spec for ComfyUI-RUM.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    print(f"Checking ComfyUI-RUM at: {ROOT}")
    for package in ("torch", "diffusers", "transformers", "safetensors", "numpy", "PIL"):
        module = __import__(package)
        print(f"{package}: {getattr(module, '__version__', 'ok')}")

    from diffusers import Flux2KleinPipeline  # noqa: F401
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel  # noqa: F401
    print("Flux2 diffusers imports: OK")

    module = load_node_module()
    print("Node classes:", ", ".join(sorted(module.NODE_CLASS_MAPPINGS)))

    config_path = ROOT / "local_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("base_model_path", "rum_checkpoint_path", "sdxl_model_path"):
            value = Path(config.get(key, ""))
            print(f"{key}: {'OK' if value.exists() else 'MISSING'} {value}")
    else:
        print("local_config.json: not found; node will use built-in defaults or manually entered paths.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
