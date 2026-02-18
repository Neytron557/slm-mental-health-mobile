# Android Deployment Status

_Updated: 2025-11-08_

## What Was Done
- Created the `android_deployment/` workspace to isolate all mobile artifacts and documentation from the rest of the repo.
- Authored `README.md` with the full ORT GenAI → device → APK playbook, matching the requested 19-step plan.
- Added `config/models.yml` to track every release checkpoint, quantization type, preferred backend, and notes for FP16 re-merging.
- Implemented automation scripts:
  - `scripts/convert_models.py` to drive ORT GenAI conversion with optional Optimum/ONNX fallback and manifest tracking.
  - `scripts/stage_models.py` to push artifacts + prompts to `/sdcard/Models/...`.
  - `scripts/run_benchmarks.py` to wrap the CLI benchmark over CPU/Vulkan/NNAPI and pull JSON metrics.
  - `scripts/aggregate_results.py` to consolidate results into CSV/Markdown leaderboards.
- Added shared assets: `templates/uniform_chat.txt` and `prompts/mental_health_eval.txt`.

## What Could Not Be Done (Yet) & Why
- **Actual conversions/builds**: ORT GenAI/ONNX exports and ORT Mobile builds were not run because they require large external repositories, GPU/NDK toolchains, and significant runtime that are out of scope for prep work in this environment.
- **Native CLI + APK binaries**: The codebase does not yet contain the C++ benchmark or Android Studio project; both depend on the ONNX Runtime builds and would introduce large binaries unsuitable for this initial scaffolding pass.
- **On-device benchmarking**: Needs a connected Galaxy S22 with USB debugging enabled plus the compiled CLI binary; not available here.

## How To Proceed
1. Install Android SDK/NDK + ORT dependencies as described in `README.md`, verify `adb devices` with the S22.
2. Merge or regenerate FP16 checkpoints for quantized models (those flagged with `needs_fp16_source: true`).
3. Run `scripts/convert_models.py` (start with Gemma + TinyLlama) and confirm manifests appear in `android_deployment/artifacts/`.
4. Stage artifacts to the phone via `scripts/stage_models.py`, then push ORT Mobile libs + CLI binary to `/data/local/tmp/slm_bench`.
5. Execute `scripts/run_benchmarks.py` across CPU/Vulkan/NNAPI; inspect the JSON metrics and aggregate via `scripts/aggregate_results.py`.
6. Once the CLI metrics look sane, clone the Android app scaffold under `android_deployment/app/` and begin wrapping the native core for the minimal Compose UI.

Document new findings or blockers here at the end of each work session so the team can resume quickly.
