package com.arm.learningpath.texttotext.inference

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.mock.MockRuntimeRunner
import com.arm.learningpath.texttotext.inference.models.tinyllama.OnnxTextGenerationAdapter

object RuntimeRunnerFactory {
    fun create(config: ModelConfig): RuntimeRunner {
        return when (config.runtime) {
            "onnxruntime" -> OnnxTextGenerationAdapter()
            "mock" -> MockRuntimeRunner()
            else -> MockRuntimeRunner(
                warning = "No adapter is registered for runtime '${config.runtime}'. Running mock output."
            )
        }
    }
}
