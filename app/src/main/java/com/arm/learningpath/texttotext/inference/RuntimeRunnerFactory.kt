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
            ModelConfig.RUNTIME_EXECUTORCH -> ExecuTorchTextGenerationAdapter()
            ModelConfig.RUNTIME_LITERT_LM -> LiteRtLmTextGenerationAdapter()
            ModelConfig.RUNTIME_LITERT,
            ModelConfig.RUNTIME_TFLITE -> LiteRtEmbeddingAdapter()
            ModelConfig.RUNTIME_ONNX,
            ModelConfig.RUNTIME_ONNX_RUNTIME,
            ModelConfig.RUNTIME_ONNX_RUNTIME_GENAI -> OnnxTextGenerationAdapter()
            // Keep mock explicit so intentional mock catalog entries run without a missing-adapter warning.
            ModelConfig.RUNTIME_MOCK -> MockRuntimeRunner()
            else -> MockRuntimeRunner(
                warning = "No adapter is registered for runtime '${config.runtime}'. Running mock output."
            )
        }
    }
}
