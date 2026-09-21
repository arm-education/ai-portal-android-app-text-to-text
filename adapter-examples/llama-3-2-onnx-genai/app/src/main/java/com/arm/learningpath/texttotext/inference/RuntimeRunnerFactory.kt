package com.arm.learningpath.texttotext.inference

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.mock.MockRuntimeRunner
import com.arm.learningpath.texttotext.inference.models.llama32.OnnxTextGenerationAdapter

object RuntimeRunnerFactory {
    fun create(config: ModelConfig): RuntimeRunner {
        return when (config.runtime) {
            ModelConfig.RUNTIME_ONNX_RUNTIME -> OnnxTextGenerationAdapter()
            ModelConfig.RUNTIME_ONNX_RUNTIME_GENAI -> OnnxTextGenerationAdapter()
            // Keep mock explicit so intentional mock catalog entries run without a missing-adapter warning.
            ModelConfig.RUNTIME_MOCK -> MockRuntimeRunner()
            else -> MockRuntimeRunner(
                warning = "No adapter is registered for runtime '${config.runtime}'. Running mock output."
            )
        }
    }
}
