# Qwen llama.cpp Example

Validated model:

```text
Arm/qwen3-5-0-8b-q4-k-m-llamacpp-vivo-x300
```

This example provides the minimal source files needed to apply the validated llama.cpp text-generation integration to the starter app.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/models/qwen/LlamaCppTextGenerationAdapter.kt`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/RuntimeRunnerFactory.kt`
- `app/src/main/AndroidManifest.xml`
- `app/build.gradle.kts`

## Download the model

From your model workspace, download the complete GGUF model package:

```bash
hf download Arm/qwen3-5-0-8b-q4-k-m-llamacpp-vivo-x300 --local-dir ./model
```

Keep the downloaded filenames and directory structure unchanged. The validated catalog expects:

```text
Qwen__Qwen3.5-0.8B_llamacpp_optimized.gguf
config.yaml
```

## Important

This example does not include:

- `app/libs/lib-release.aar`
- GGUF model files
- build outputs or APKs

Before building this example, build the official llama.cpp Android library from the Learning Path and place the generated AAR at:

```text
app/libs/lib-release.aar
```

## Apply

From the starter app root:

```bash
cp -R adapter-examples/qwen-llamacpp/app/. app/
```

After adding the official AAR, build the app from Android Studio or run:

```bash
./gradlew assembleDebug
```

Copy the GGUF model package files listed in `model_catalog.json` to:

```text
filesDir/models/qwen3-5-0-8b-q4-k-m-llamacpp-vivo-x300/
```
