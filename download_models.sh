#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

hf download black-forest-labs/FLUX.2-klein-base-4B \
  model_index.json \
  --local-dir "$MODELS_DIR/FLUX.2-klein-base-4B" \
  --max-workers 4

hf download black-forest-labs/FLUX.2-klein-base-4B \
  --local-dir "$MODELS_DIR/FLUX.2-klein-base-4B" \
  --include 'scheduler/*' 'text_encoder/*' 'tokenizer/*' 'transformer/*' 'vae/*' \
  --max-workers 8

hf download rimochan/RUM-FLUX.2-klein-4B-preview \
  model-checkpoint-608000.safetensors \
  --local-dir "$MODELS_DIR/RUM-FLUX.2-klein-4B-preview" \
  --max-workers 4

hf download Ine007/waiIllustriousSDXL_v160 \
  --local-dir "$MODELS_DIR/waiIllustriousSDXL_v160_text" \
  --include 'model_index.json' 'text_encoder/*' 'text_encoder_2/*' 'tokenizer/*' 'tokenizer_2/*' \
  --max-workers 8

cat <<EOF

Downloaded models:
  base_model_path=$MODELS_DIR/FLUX.2-klein-base-4B
  rum_checkpoint_path=$MODELS_DIR/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors
  sdxl_model_path=$MODELS_DIR/waiIllustriousSDXL_v160_text
EOF
