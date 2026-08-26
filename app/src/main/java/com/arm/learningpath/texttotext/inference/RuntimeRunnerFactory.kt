package com.arm.learningpath.texttotext.inference

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.embedding.LiteRtEmbeddingAdapter
import com.arm.learningpath.texttotext.inference.generation.ExecuTorchTextGenerationAdapter
import com.arm.learningpath.texttotext.inference.generation.LiteRtLmTextGenerationAdapter
import com.arm.learningpath.texttotext.inference.generation.OnnxTextGenerationAdapter
import com.arm.learningpath.texttotext.inference.mock.MockRuntimeRunner

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
