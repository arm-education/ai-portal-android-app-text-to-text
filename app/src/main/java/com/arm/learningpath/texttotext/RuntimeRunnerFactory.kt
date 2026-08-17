package com.arm.learningpath.texttotext

object RuntimeRunnerFactory {
    fun create(config: ModelConfig): RuntimeRunner {
        return when (config.runtime) {
            "executorch" -> ExecuTorchTextGenerationAdapter()
            "litert-lm" -> LiteRtLmTextGenerationAdapter()
            "litert", "tflite" -> LiteRtEmbeddingAdapter()
            "onnx", "onnxruntime", "onnxruntime-genai" -> OnnxTextGenerationAdapter()
            "mock" -> MockRuntimeRunner()
            else -> MockRuntimeRunner(
                warning = "No adapter is registered for runtime '${config.runtime}'. Running mock output."
            )
        }
    }
}
