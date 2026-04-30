from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_comfy_root = ROOT.parent.parent
    parser = argparse.ArgumentParser(description="Check a ComfyUI-RUM native installation.")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=default_comfy_root,
        help="ComfyUI root directory. Default: inferred from custom_nodes/ComfyUI-RUM/scripts.",
    )
    return parser.parse_args()


def load_node_module(comfy_root: Path):
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
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
    args = parse_args()
    comfy_root = args.comfy_root.expanduser().resolve()
    print(f"Checking ComfyUI-RUM at: {ROOT}")
    print(f"ComfyUI root: {comfy_root}")
    for package in ("torch", "safetensors"):
        module = __import__(package)
        print(f"{package}: {getattr(module, '__version__', 'ok')}")

    module = load_node_module(comfy_root)
    print("Node classes:", ", ".join(sorted(module.NODE_CLASS_MAPPINGS)))

    import folder_paths

    for folder_name in ("diffusion_models", "text_encoders", "vae"):
        try:
            names = folder_paths.get_filename_list(folder_name)
        except Exception as exc:
            print(f"{folder_name}: ERROR {exc}")
            continue
        print(f"{folder_name}: {len(names)} file(s) visible to ComfyUI")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
