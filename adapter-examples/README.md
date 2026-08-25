# Validated Adapter Examples

This directory contains minimal, validated integration files for selected text-to-text Android models.

The starter app stays lightweight by default. It includes mock catalog entries and runtime adapter stubs, but it does not enable every runtime dependency in the APK. To run one of the validated examples, copy only that example's files into the starter app, build the app, then copy the model artifacts to app-local storage as described in the Learning Path.

Each example includes:

- `app/src/main/assets/model_catalog.json`
- one matching runtime adapter Kotlin file
- `app/build.gradle.kts` from the validated integration
- a short model-specific README

Each example excludes:

- downloaded model artifacts
- local AAR files
- APKs and build outputs
- `.gradle`, `.kotlin`, `.idea`, and local cache files
- device-specific files such as `local.properties`

The generic adapter-agent prompt in the Learning Path remains the fallback for any model that does not match these examples.

## Apply an example

From the starter app root, copy one example into the project:

```bash
cp -R adapter-examples/<example-name>/app/. app/
```

Then open or sync the project in Android Studio and build the app.

Use only one example at a time. If you switch to another runtime, reapply the clean starter app or undo the previous example before applying the next one.

