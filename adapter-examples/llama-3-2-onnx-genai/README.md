# Llama 3.2 ONNX Runtime GenAI Example

Validated model:

```text
Arm/llama-3-2-1b-instruct-onnx-genai-int4-kquantlast-emb-int8-vivo-x300
```

This example provides the minimal source files needed to apply the validated ONNX Runtime GenAI text-generation integration to the starter app.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/models/llama32/OnnxTextGenerationAdapter.kt`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/RuntimeRunnerFactory.kt`
- `app/build.gradle.kts`

## Download the model

From your model workspace, download the complete ONNX Runtime GenAI model directory:

```bash
hf download Arm/llama-3-2-1b-instruct-onnx-genai-int4-kquantlast-emb-int8-vivo-x300 --local-dir ./model
```

Keep the downloaded filenames and directory structure unchanged.

## Important

This example does not include:

- `app/libs/onnxruntime-genai-release.aar`
- ONNX model files
- tokenizer files
- external data files
- build outputs or APKs

Before building this example, complete the ONNX Runtime GenAI prerequisite from the Learning Path and place the official AAR at:

```text
app/libs/onnxruntime-genai-release.aar
```

## Apply

From the starter app root:

```bash
cp -R adapter-examples/llama-3-2-onnx-genai/app/. app/
```

After adding the official AAR, build the app from Android Studio or run:

```bash
./gradlew assembleDebug
```

Copy the ONNX Runtime GenAI model directory files listed in `model_catalog.json` to:

```text
filesDir/models/llama-3-2-1b-instruct-onnx-genai-int4-kquantlast-emb-int8-vivo-x300/
```
