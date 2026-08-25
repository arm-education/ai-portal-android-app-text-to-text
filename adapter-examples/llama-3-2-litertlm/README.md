# Llama 3.2 LiteRT-LM Example

Validated model:

```text
Arm/llama-3-2-1b-instruct-litertlm-int8
```

This example provides the minimal files needed to apply the validated LiteRT-LM text-generation integration to the starter app.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/LiteRtLmTextGenerationAdapter.kt`
- `app/build.gradle.kts`

## Apply

From the starter app root:

```bash
cp -R adapter-examples/llama-3-2-litertlm/app/. app/
```

Then build the app from Android Studio or run:

```bash
./gradlew assembleDebug
```

Copy the model files listed in `model_catalog.json` to:

```text
filesDir/models/llama-3-2-1b-instruct-litertlm-int8/
```

The `.litertlm` bundle, tokenizer files, cache files, and downloaded model artifacts are not included in this example.

