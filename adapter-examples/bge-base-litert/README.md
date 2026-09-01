# BGE Base LiteRT Embedding Example

Validated model:

```text
Arm/bge-base-en-v1.5-int8-litert
```

This example provides the minimal files needed to apply the validated LiteRT text-embedding integration to the starter app.

## Download model files

Download the LiteRT model and configuration files from the Arm repository:

```bash
hf download Arm/bge-base-en-v1.5-int8-litert --local-dir ./model
```

Download the tokenizer files from the base model repository:

```bash
hf download BAAI/bge-base-en-v1.5 \
  --include "tokenizer*" "vocab.txt" "special_tokens_map.json" \
  --local-dir ./model
```

The `Arm/bge-base-en-v1.5-int8-litert` repository provides the LiteRT model and configuration files. The `BAAI/bge-base-en-v1.5` repository provides the tokenizer files needed by the validated adapter.

## Files

- `app/src/main/assets/model_catalog.json`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/models/bge/LiteRtEmbeddingAdapter.kt`
- `app/src/main/java/com/arm/learningpath/texttotext/inference/RuntimeRunnerFactory.kt`
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
