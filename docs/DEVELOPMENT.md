# Development Notes

## Local Setup

```bash
python -m pip install -r requirements.txt
python scripts/check_install.py
```

## Model Files

Use the helper:

```bash
python scripts/download_models.py --write-local-config
```

or create `local_config.json` manually from `local_config.example.json`.

## Architecture

The node uses a diffusers pipeline wrapper rather than ComfyUI's native model/sampler stack.

Main files:

- `nodes.py`: ComfyUI node definitions and UI inputs.
- `rum_diffusers.py`: RUM FLUX.2-klein transformer and pipeline adapter.
- `scripts/download_models.py`: model downloader.
- `scripts/check_install.py`: import and path validator.

## Upstream References

The adapter mirrors the upstream RUM inference path:

- `推理.py`
- `哭.py`
- `train_flux_klein哭.py`

The key behavior is concatenating FLUX.2-klein prompt embeddings with SDXL CLIP embeddings before calling the modified transformer.
