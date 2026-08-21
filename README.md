# Text to Text Android Starter

Generic Android starter app for running text-generation and text-embedding models in the Arm text-to-text Learning Path.

This app is used by the Arm Learning Path: **Run text-to-text models on Android**. It is intended for learning how text-to-text models run on devices and is not a reference production application. It is provided under the [Arm Education End User License Agreement](LICENSE.md).

The app is intentionally model-independent. It provides:

- a model selector loaded from `app/src/main/assets/model_catalog.json`
- app-local model path validation under `filesDir/models/<model-id>/`
- a prompt/text input
- load and run buttons
- result and timing display
- runtime adapter stubs for ExecuTorch, LiteRT-LM, LiteRT embeddings, and ONNX
- a mock runner so the UI can be checked before real runtime integration

## Open the project

Open this directory in Android Studio:

```text
text-to-text-android-starter
```

Sync the Gradle project, connect an Android device, and run the app.

The starter app pins Java and Kotlin compilation to JVM 17 in `app/build.gradle.kts`.

The default mock catalog entries do not require model files. Use them to confirm that the app opens, loads, and runs before you add a real runtime adapter. When a mock entry is selected, the app ignores `filesDir/models/<model-id>/`.

## Add a model

Edit `app/src/main/assets/model_catalog.json` and add one entry for your selected model. Keep model-specific fields in the catalog instead of hard-coding them in the UI.

Before running the inspection script, make sure the local model folder contains the runtime artifact and the tokenizer files required by the model card. Some runtime artifact repositories reference a tokenizer from a base model repository instead of packaging those files directly. In that case, download the tokenizer files from the base model repository into the same local model folder first.

You can generate local context for a downloaded model folder:

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

Use those files with the adapter-agent prompt shown in the Learning Path page.

Model files are loaded from:

```text
/data/data/com.arm.learningpath.texttotext/files/models/<model-id>/
```

For development, copy files through `adb` and `run-as` after the app is installed once:

```bash
adb shell run-as com.arm.learningpath.texttotext mkdir -p files/models/<model-id>
adb push <local-model-file> /data/local/tmp/<model-file>
adb shell run-as com.arm.learningpath.texttotext cp /data/local/tmp/<model-file> files/models/<model-id>/<model-file>
```

For directory artifacts, create the same directory structure under `files/models/<model-id>/`.

## Complete a runtime adapter

Use the adapter-agent prompt shown in the Learning Path page with an approved AI coding agent. The prompt asks the agent to read the model card, inspected model context, and local starter app, then edit only the selected runtime adapter and Gradle dependency files.

Runtime tensor names are not always semantic. If an exported Android artifact exposes internal, numeric, or graph-generated tensor names, map semantic inputs using the model card, runtime config, tokenizer/config files, documented order, shapes, and dtypes. Do not fail only because a runtime tensor name differs from a model-card name.

Do not add Hugging Face tokens, Arm AI Portal credentials, or long-lived artifact credentials to this app.

## License

This project is provided under the [Arm Education End User License Agreement](LICENSE.md).
