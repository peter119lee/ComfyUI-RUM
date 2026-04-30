# Changelog

## 0.2.5

- Documented why native ComfyUI cannot exactly reproduce the old diffusers output.
- Added direct model-source mapping, Windows/Aki download commands, and workflow-specific model requirements to README.
- Added `--include-teacher-clip` and `--all` to the model download helper.
- Expanded install checks so missing sample-workflow model files are visible immediately.
- Added model requirement notes/titles to sample workflows.

## 0.2.4

- Fixed the visual diffusers-match workflow so it uses the same native approximation path as the API workflow.
- Routed the positive Qwen encode through the diffusers-match layer override while keeping negative on default 512-token Qwen conditioning.
- Made `RUM FLUX.2 Set Qwen Layers` safer by temporarily applying the layer override only during encoding and forcing cache invalidation.
- Rewrote README in Simplified Chinese and documented normal ComfyUI model paths.

## 0.2.3

- Fixed diffusers-match token cropping when ComfyUI pads FLUX.2 text conditioning to 512 tokens.
- Updated example workflows to use Windows-compatible nested diffusion model paths.

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
