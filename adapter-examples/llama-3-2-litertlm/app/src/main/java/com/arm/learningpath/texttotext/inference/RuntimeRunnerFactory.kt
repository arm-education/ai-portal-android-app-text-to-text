package com.arm.learningpath.texttotext.inference

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.mock.MockRuntimeRunner
import com.arm.learningpath.texttotext.inference.models.llama32.LiteRtLmTextGenerationAdapter

object RuntimeRunnerFactory {
    fun create(config: ModelConfig): RuntimeRunner {
        return when (config.runtime) {
            ModelConfig.RUNTIME_LITERT_LM -> LiteRtLmTextGenerationAdapter()
            // Keep mock explicit so intentional mock catalog entries run without a missing-adapter warning.
            ModelConfig.RUNTIME_MOCK -> MockRuntimeRunner()
            else -> MockRuntimeRunner(
                warning = "No adapter is registered for runtime '${config.runtime}'. Running mock output."
            )
        }
    }
}
