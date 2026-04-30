# Changelog

## 0.2.2

- Added `RUM FLUX.2 Diffusers Match Model Patch` for old diffusers-style token cropping.
- Added `examples/diffusers_match_workflow_api.json` with the old example seed and `base_text_tokens=200`.

## 0.2.1

- Updated the recommended native workflow to use `Flux2Scheduler` + `SamplerCustomAdvanced` instead of plain `KSampler scheduler=simple`.
- Documented the noisy/glitched image failure mode caused by the wrong FLUX.2 sampling chain.

## 0.2.0

- Switched `main` to native ComfyUI adapter mode.
- Added `RUM FLUX.2 Apply Model Patch`.
- Added `RUM FLUX.2 Combine Conditioning`.
- Moved the old diffusers pipeline implementation to the `diffusers-pipeline` branch.

## 0.1.0

- Initial diffusers pipeline wrapper.
