# BGE Base LiteRT Embedding Example

Validated model:

```text
Arm/bge-base-en-v1.5-int8-litert
```

This example provides the minimal files needed to apply the validated LiteRT text-embedding integration to the starter app.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/LiteRtEmbeddingAdapter.kt`
- `app/build.gradle.kts`

## Apply

From the starter app root:

```bash
cp -R adapter-examples/bge-base-litert/app/. app/
```

Then build the app from Android Studio or run:

```bash
./gradlew assembleDebug
```

Copy the model and tokenizer files listed in `model_catalog.json` to:

```text
filesDir/models/bge-base-en-v1.5-int8-litert/
```

The `.tflite` files and tokenizer files are not included in this example. If the runtime artifact repository does not include tokenizer files, download the tokenizer files from the base model repository as described in the Learning Path.

