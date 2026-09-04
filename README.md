# Text to Text Android Starter

This repository contains an Android starter application for running text-generation and text-embedding models as part of the Arm Learning Path **Run text-to-text models on Android**.

The app is intended for learning how text-to-text models run on the device and is not a reference production application. It is provided under the [Arm Education End User License Agreement](LICENSE.md).

The base app declares no Android permissions and does not send prompts or results off the device.

The app is intentionally model-independent. It provides:

- a model selector loaded from `app/src/main/assets/model_catalog.json`
- app-private model directory resolution under `filesDir/models/<model-id>/`
- a prompt or text input
- load and run buttons
- result and timing display
- runtime adapter stubs for ExecuTorch text generation, LiteRT-LM text generation, LiteRT text embedding, and ONNX Runtime text generation
- a mock runner so the UI can be checked before real runtime integration

Validated [adapter examples](adapter-examples/) are provided when your selected model matches a validated integration. Keep the default app lightweight until you choose a runtime.

## Project structure

The app separates reusable infrastructure from runtime and model-specific code:

```text
app/src/main/java/com/arm/learningpath/texttotext/
├── ui/
├── catalog/
├── storage/
└── inference/
    ├── embedding/
    ├── generation/
    └── mock/
```

The reusable shell lives in:

- `ui/` for the Android activity and shared text input/result interface.
- `catalog/` for catalog parsing and model metadata.
- `storage/` for app-private model directory resolution.
- `inference/` for the runtime adapter interface, result contract, and factory registration.

The base app keeps mock and placeholder adapters under `inference/mock/`, `inference/generation/`, and `inference/embedding/`. Validated examples place model-specific implementations under `inference/models/<model-name>/` and replace `inference/RuntimeRunnerFactory.kt` so the selected catalog runtime maps to that implementation.

For a new model, developers should normally update `app/src/main/assets/model_catalog.json`, add or adapt a model-specific package under `inference/models/`, and register that adapter in `inference/RuntimeRunnerFactory.kt`. Keep tokenizer handling, tensor mapping, preprocessing, runtime execution, and output cleanup out of the UI layer so each part can be reviewed and tested independently.

## Requirements

The base app requires:

- Android Studio with Android SDK Platform 35
- JDK 17
- an Android device or emulator running Android 8.0 (API level 26) or later

Python 3.10 or later is required only for `scripts/inspect_android_model.py`. Android SDK Platform-Tools (`adb`) are required only to copy model artifacts manually. Adapter examples can set a higher minimum API level or add runtime-specific requirements; review an example's README and `app/build.gradle.kts` before applying it.

## Open the project

Open the repository root in Android Studio. Sync the Gradle project, connect an Android device or start an emulator, and run the app.

The default mock catalog entries do not require model files. Use them to confirm that the app opens, loads, and runs before you add a real runtime adapter. When a mock entry is selected, the app ignores `filesDir/models/<model-id>/`.

Run the local unit tests to check catalog parsing, runtime routing, and mock runner behavior:

```bash
./gradlew testDebugUnitTest
```

## Add a model

Edit `app/src/main/assets/model_catalog.json` and add one entry for your selected model. Keep model-specific fields in the catalog instead of hard-coding them in the UI.

Before running the inspection script, make sure the local model folder contains the runtime artifact and the tokenizer files required by the model card. Some runtime artifact repositories reference a tokenizer from a base model repository instead of packaging those files directly. In that case, download the tokenizer files from the base model repository into the same local model folder first.

From the repository root, generate local context for a downloaded model folder:

```bash
python3 scripts/inspect_android_model.py \
  --model-id <model-id> \
  --model-source <model-source-or-card-url> \
  --runtime <executorch|litert|litert-lm|onnxruntime> \
  --workload <text-generation|text-embedding> \
  --local-model-dir <local-model-folder>
```

The script writes:

- `android_model_config.json`
- `model-context/model-summary.json`
- small copied metadata files under `model-context/metadata`

Each run overwrites `android_model_config.json` and replaces `model-context/` and its contents.

Use those files with one of the validated adapter examples, or with the adapter-agent prompt shown in the Learning Path page when your selected model does not match an example.

The app resolves model files under its app-private files directory:

```text
filesDir/models/<model-id>/
```

For development, install a debug build once, then copy files through `adb` and `run-as`:

```bash
adb shell run-as com.arm.learningpath.texttotext mkdir -p files/models/<model-id>
adb push <local-model-file> /data/local/tmp/<model-file>
adb shell run-as com.arm.learningpath.texttotext cp /data/local/tmp/<model-file> files/models/<model-id>/<model-file>
```

After confirming that the copy succeeded, remove the temporary staging file:

```bash
adb shell rm /data/local/tmp/<model-file>
```

For directory artifacts, create the same directory structure under `files/models/<model-id>/`.

Clearing the app's storage or uninstalling the app removes files from its app-private directory. The app does not otherwise delete imported model files.

## Complete a runtime adapter

Use the adapter-agent prompt shown in the Learning Path page with an approved AI coding agent. The prompt asks the agent to read the model card, inspected model context, and local starter app, then update the selected runtime adapter, catalog entry, factory registration, and Gradle dependency files needed for that model.

Runtime tensor names are not always semantic. If an exported Android artifact exposes internal, numeric, or graph-generated tensor names, map semantic inputs using the model card, runtime config, tokenizer/config files, documented order, shapes, and dtypes. Do not fail solely because a runtime tensor name differs from a model-card name.

Do not add Hugging Face tokens, Arm AI Portal credentials, or long-lived artifact credentials to this app.

## License

This project is provided under the [Arm Education End User License Agreement](LICENSE.md).
