# ComfyUI-RUM

ComfyUI custom nodes for [RimoChan/RUM](https://github.com/RimoChan/RUM), focused on the public `RUM-FLUX.2-klein-4B-preview` checkpoint.

RUM is a cross-architecture distillation experiment that transfers anime generation behavior from an SDXL teacher model into newer architectures. This node wraps the upstream diffusers inference path so RUM can be used inside ComfyUI.

## What It Does

- Loads the FLUX.2-klein base model.
- Loads the RUM FLUX.2-klein preview checkpoint.
- Loads SDXL text encoders/tokenizers for the extra RUM CLIP conditioning branch.
- Generates ComfyUI `IMAGE` output from a prompt.
- Provides an unload node to clear cached RUM pipelines.

## What It Does Not Do Yet

This is a self-contained diffusers pipeline node, not a native ComfyUI sampler stack.

Current limitations:

- No KSampler-compatible model object.
- No latent input/output workflow.
- No ControlNet/IP-Adapter support.
- No LoRA stacking through ComfyUI's normal loader nodes.
- No negative prompt input yet.

## Nodes

- `RUM FLUX.2-klein Loader`
- `RUM FLUX.2-klein Sampler`
- `RUM Unload Models`

## Installation

Clone this repo into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

Restart ComfyUI.

> Important: install dependencies into the same Python environment that runs ComfyUI.

## Model Download

The required files are large, about 24 GB total when using the text-only SDXL teacher folder.

```bash
python scripts/download_models.py --write-local-config
```

This downloads:

| Purpose | Source | Default folder |
| --- | --- | --- |
| FLUX.2-klein base | `black-forest-labs/FLUX.2-klein-base-4B` | `models/FLUX.2-klein-base-4B` |
| RUM checkpoint | `rimochan/RUM-FLUX.2-klein-4B-preview` | `models/RUM-FLUX.2-klein-4B-preview` |
| SDXL text encoders | `Ine007/waiIllustriousSDXL_v160` | `models/waiIllustriousSDXL_v160_text` |

The script writes `local_config.json` when `--write-local-config` is used. That file is intentionally ignored by Git.

## Manual Model Paths

If you already have the files, create `local_config.json` beside `nodes.py`:

```json
{
  "base_model_path": "models/FLUX.2-klein-base-4B",
  "rum_checkpoint_path": "models/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors",
  "sdxl_model_path": "models/waiIllustriousSDXL_v160_text"
}
```

You can also paste absolute paths directly into the loader node.

Environment variables are supported:

- `RUM_BASE_MODEL_PATH`
- `RUM_CHECKPOINT_PATH`
- `RUM_SDXL_MODEL_PATH`

## Basic Workflow

1. Add `RUM FLUX.2-klein Loader`.
2. Add `RUM FLUX.2-klein Sampler`.
3. Connect `pipeline` from the loader to the sampler.
4. Connect sampler `images` to `Preview Image` or `Save Image`.

A minimal API workflow is provided at:

```text
examples/basic_workflow_api.json
```

Suggested first settings:

```text
dtype=bfloat16
device=auto
sdxl_text_device=cpu
unload_sdxl_unet_vae=true
steps=20
guidance_scale=5
width=960
height=1024
max_sequence_length=200
text_encoder_out_layers=10,20,30
```

If VRAM is tight, reduce width/height first.

## Verify Installation

Run this in the custom node directory using ComfyUI's Python:

```bash
python scripts/check_install.py
```

It verifies:

- required Python packages
- `Flux2KleinPipeline` availability
- custom node import
- optional `local_config.json` model paths

## Why Stock ComfyUI FLUX Loading Is Not Enough

RUM modifies FLUX.2-klein's transformer to consume two text-conditioning streams:

1. normal FLUX.2-klein text embeddings
2. extra SDXL CLIP embeddings from the anime teacher model

Stock ComfyUI FLUX nodes do not provide the second branch, so the RUM checkpoint cannot be loaded as a normal FLUX checkpoint without additional adapter code.

## Troubleshooting

### `No file named model_index.json`

Your FLUX.2-klein base folder is incomplete. Run:

```bash
python scripts/download_models.py --write-local-config
```

or make sure `base_model_path` points to a full diffusers folder containing `model_index.json`.

### `cannot import name Flux2KleinPipeline`

Upgrade diffusers in ComfyUI's Python environment:

```bash
python -m pip install --upgrade "diffusers>=0.37.1"
```

### CUDA out of memory

Try:

- `sdxl_text_device=cpu`
- lower width/height
- restart ComfyUI
- run `RUM Unload Models`

### Slow first generation

This is expected. The node loads FLUX.2-klein base weights, the RUM checkpoint, and SDXL text encoders.

## Credits

- RUM method and original training/inference scripts: [RimoChan/RUM](https://github.com/RimoChan/RUM)
- FLUX.2-klein base model: `black-forest-labs/FLUX.2-klein-base-4B`
- RUM preview weights: `rimochan/RUM-FLUX.2-klein-4B-preview`
- SDXL teacher text encoders used by the download helper: `Ine007/waiIllustriousSDXL_v160`

## License / Upstream Permission

This repository includes adapter code derived from the public RUM inference implementation. The upstream RUM repository did not include a license when this wrapper was created, so public redistribution should be treated as pending upstream permission unless a license is later added or explicit permission is granted.

Model files are not included and remain under their respective upstream licenses.
