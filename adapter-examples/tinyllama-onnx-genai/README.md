# TinyLlama ONNX Runtime GenAI Example

Validated model:

```text
Arm/tinyllama-1-1b-chat-onnx-genai-int4-kquantlast-emb-int8-vivo-x300
```

This example provides the minimal source files needed to apply the validated ONNX Runtime GenAI text-generation integration to the starter app.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/models/tinyllama/OnnxTextGenerationAdapter.kt`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/RuntimeRunnerFactory.kt`
- `app/build.gradle.kts`

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
cp -R adapter-examples/tinyllama-onnx-genai/app/. app/
```

After adding the official AAR, build the app from Android Studio or run:

```bash
./gradlew assembleDebug
```

Copy the ONNX Runtime GenAI model directory files listed in `model_catalog.json` to:

```text
filesDir/models/tinyllama-1-1b-chat-onnx-genai-int4-kquantlast-emb-int8-vivo-x300/
```

