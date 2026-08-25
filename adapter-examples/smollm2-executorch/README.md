# SmolLM2 ExecuTorch Example

Validated model:

```text
Arm/smollm2-360m-instruct-8da4w-xnnpack-executorch
```

This example provides the minimal files needed to apply the validated ExecuTorch text-generation integration to the starter app.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/ExecuTorchTextGenerationAdapter.kt`
- `app/build.gradle.kts`

## Apply

From the starter app root:

```bash
cp -R adapter-examples/smollm2-executorch/app/. app/
```

Then build the app from Android Studio or run:

```bash
./gradlew assembleDebug
```

Copy the model files listed in `model_catalog.json` to:

```text
filesDir/models/smollm2-360m-instruct-8da4w-xnnpack-executorch/
```

The model artifact and tokenizer files are not included in this example.

