# Android Deployment Pipeline

This directory contains everything required to convert the SLM release checkpoints to ONNX Runtime Mobile/GenAI, push them to a Samsung Galaxy S22 (Snapdragon 8 Gen 1), and benchmark them across CPU, Vulkan, and NNAPI execution providers before wrapping the native core into a minimal APK UI.

## High-Level Flow
- **Normalize checkpoints** inside `intern_summer/models/release_models` so that each model has an FP16 baseline (merge LoRA adapters where needed). Quantized-only folders (*bnb4, *awq, *gptq*) should either be re-merged or flagged for the fallback ONNX path.
- **Convert with ORT GenAI** (primary) via `scripts/convert_models.py`; automatically copies tokenizer assets and records metadata in `artifacts/<model>/manifest.json`.
- **Fallback to standard ONNX** when ORT GenAI cannot handle a model (e.g., unsupported GPTQ layout). The same script can trigger the Optimum + ORT dynamic quantization path.
- **Stage artifacts** to `/sdcard/Models/<model_name>` using `scripts/stage_models.py`, ensuring each directory contains `model.ort` (or decoder ONNX), tokenizer assets, and optional `generation_config.json`.
- **Build ORT Mobile** (CPU+Vulkan+NNAPI) and the native CLI benchmark following `# Build Artifacts` below. Push libs/binaries to `/data/local/tmp/slm_bench`.
- **Benchmark** with `scripts/run_benchmarks.py`, which wraps `adb shell` invocations of the CLI and gathers JSON metrics (load time, TTFT, decode time, tokens/s, VmHWM) for each model/EP/repeat.
- **Aggregate results** via `scripts/aggregate_results.py` to produce CSV/Markdown summaries. Later, reuse the same native core inside the Compose-based APK UI.

## Directory Layout

```
android_deployment/
├── README.md
├── DEPLOYMENT_STATUS.md          # Filled in per sprint; explains progress & blockers
├── config/
│   └── models.yml                # Registry of release checkpoints and conversion knobs
├── prompts/
│   └── mental_health_eval.txt    # Shared prompt set pushed to /sdcard/Models/prompts.txt
├── templates/
│   └── uniform_chat.txt          # Single chat template applied across all models
└── scripts/
    ├── convert_models.py         # ORT GenAI conversion with ONNX fallback
    ├── stage_models.py           # adb push helper for artifacts & prompts
    ├── run_benchmarks.py         # Repeated CLI benchmarking over EP/model grid
    └── aggregate_results.py      # Consolidate JSON metrics into CSV
```

> **Add new automation here.** All new files, logs, or helper notes for the mobile track should be scoped to this directory to keep the rest of the repo untouched.

## Environment Setup

1. **Android SDK/NDK & Tooling**
   ```bash
   sdkmanager --install \
     "platform-tools" \
     "platforms;android-34" \
     "build-tools;34.0.0" \
     "ndk;26.1.10909125" \
     "cmake;3.22.1"
   ```
   Export:
   ```bash
   export ANDROID_SDK_ROOT=~/Android/Sdk
   export ANDROID_NDK_ROOT=$ANDROID_SDK_ROOT/ndk/26.1.10909125
   export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH
   ```

2. **Python + Pip**
   ```bash
   cd intern_summer && source .venv/bin/activate
   pip install -r requirements.txt  # add ort-genai/optimum deps as needed
   ```

3. **Repositories**
   ```bash
   git clone https://github.com/microsoft/onnxruntime-genai.git
   git clone https://github.com/microsoft/onnxruntime.git
   ```

4. **ADB / Device**
   - Enable USB debugging on the Galaxy S22.
   - Verify connectivity: `adb devices`.

## Model Registry (`config/models.yml`)

- Lists every folder under `intern_summer/models/release_models`.
- Tracks quantization format, preferred precision, max sequence length, and whether fallback is allowed.
- Scripts read this file to keep automation in sync with the checkpoints that arrive during the internship.

## Conversion Workflow

### ORT GenAI (Primary)
```bash
python android_deployment/scripts/convert_models.py \
  --config android_deployment/config/models.yml \
  --artifacts-root android_deployment/artifacts \
  --backend ort-genai \
  --precision int4 \
  --max-seq-len 4096
```
- Wraps `python -m onnxruntime_genai.models.convert`.
- Automatically mirrors tokenizer assets (`tokenizer.json`, `special_tokens_map.json`, etc.).
- Stores per-model metadata (`manifest.json`) with hashes and timestamps for provenance.

### Standard ONNX (Fallback)
```bash
python android_deployment/scripts/convert_models.py \
  --backend onnx \
  --models Llama-3.2-3B-Instruct-finetuned-gptq-w4g128
```
- Uses `optimum-cli export onnx --task text-generation-with-past`.
- Applies ORT dynamic quantization (`--weight_type QInt8`) and checks graph correctness via ORT checkers.
- Suitable for GPTQ/AWQ cases where ORT GenAI lacks kernels.

Both paths output to `android_deployment/artifacts/<model>/<backend>/`.

## Templates & Prompts
- `templates/uniform_chat.txt` implements the shared chat template:
  ```
  <<SYS>>{system}<<SYS>>
  <<USER>>{user}<<ASSISTANT>>
  ```
- `prompts/mental_health_eval.txt` holds 10+ representative counseling prompts; push to `/sdcard/Models/prompts.txt` for consistent benchmarking.

## Build Artifacts

1. **ONNX Runtime Mobile**
   ```bash
   cd onnxruntime
   ./build.sh --config Release --build_shared_lib --parallel \
     --android --android_ndk $ANDROID_NDK_ROOT \
     --android_abi arm64-v8a --android_api 30 --enable_vulkan

   ./build.sh --config Release --build_shared_lib --parallel \
     --android --android_ndk $ANDROID_NDK_ROOT \
     --android_abi arm64-v8a --android_api 30 --enable_nnapi
   ```
   - Outputs `libonnxruntime.so`, `libonnxruntime_providers_vulkan.so`, `libonnxruntime_providers_nnapi.so`.

2. **ORT GenAI for Android**
   ```bash
   cd onnxruntime-genai
   ./scripts/build_android.sh --abi arm64-v8a --api 30 \
     --enable_vulkan --enable_nnapi
   ```

3. **Native CLI Benchmark**
   - Place source under `android_deployment/native/` (to be added next).
   - Use NDK toolchain + CMake/Ninja to compile against the ORT Mobile + ORT GenAI libs.
   - Implement metrics: load_ms, ttft_ms, decode_ms, tokens/s, VmHWM sampling thread.

4. **Device Staging**
   ```bash
   adb shell 'mkdir -p /data/local/tmp/slm_bench'
   adb push build/libs/arm64-v8a/*.so /data/local/tmp/slm_bench/
   adb push slm_bench /data/local/tmp/slm_bench/slm_bench
   adb shell 'chmod +x /data/local/tmp/slm_bench/slm_bench'
   ```

## Benchmark Automation

1. **Stage Models**
   ```bash
   python android_deployment/scripts/stage_models.py \
     --artifacts-root android_deployment/artifacts \
     --device-model-root /sdcard/Models \
     --adb adb
   ```

2. **Run Benchmarks**
   ```bash
   python android_deployment/scripts/run_benchmarks.py \
     --config android_deployment/config/models.yml \
     --device-binary /data/local/tmp/slm_bench/slm_bench \
     --device-lib-dir /data/local/tmp/slm_bench \
     --prompt-file /sdcard/Models/prompts.txt \
     --eps cpu vulkan nnapi --repeats 3 \
     --results-root android_deployment/results
   ```
   - Wraps `adb shell 'cd ... && LD_LIBRARY_PATH=. ./slm_bench ...'`.
   - Pulls JSON metrics (`bench_<ep>.json`) back to the host per run.

3. **Aggregate**
   ```bash
   python android_deployment/scripts/aggregate_results.py \
     --input-glob 'android_deployment/results/**/*.json' \
     --output android_deployment/results/benchmarks.csv
   ```

## Minimal APK UI (Next Stage)

Once the CLI is validated:
- Create `android_deployment/app/` Android Studio project.
- Drop native libs into `app/src/main/jniLibs/arm64-v8a/`.
- Write JNI wrappers to expose the CLI functionality (`loadModel`, `generate`).
- Build a Compose UI with prompt input, model dropdown, EP selector, and live metrics.
- Save JSON outputs to `Documents/slm_bench/results` mirroring the CLI schema.

## Risks & Mitigations
- **Quantized-only checkpoints**: Document in `DEPLOYMENT_STATUS.md`, prioritize re-merging FP16 weights. Use ONNX fallback if blocked.
- **NNAPI driver variability**: Always collect CPU/Vulkan baselines; log provider warnings to spot misconfigurations.
- **Memory pressure on 3B models**: Prefer W4A16, shorter prompts (64-128 tokens) for latency tests, and monitor VmHWM.
- **Tooling drift**: Pin SDK/NDK versions, requirements, and script dependencies in this folder to keep the rest of the repo stable.

## Deliverables Checklist
- [ ] ORT GenAI artifacts per model under `android_deployment/artifacts/`.
- [ ] `/sdcard/Models/<model>` staged with tokenizer assets + prompt file.
- [ ] `/data/local/tmp/slm_bench/` populated with libs + CLI binary.
- [ ] JSON metrics pulled into `android_deployment/results/`.
- [ ] Aggregated CSV/plots for TPM/latency/memory.
- [ ] Compose APK installed and validated on the Galaxy S22.

Update `DEPLOYMENT_STATUS.md` at the end of each iteration to capture what shipped, what is blocked, and next steps.
