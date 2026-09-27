package com.arm.learningpath.texttotext.inference

import android.content.Context
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.mock.MockRuntimeRunner
import com.arm.learningpath.texttotext.inference.models.qwen.LlamaCppTextGenerationAdapter

object RuntimeRunnerFactory {
    fun create(config: ModelConfig, context: Context? = null): RuntimeRunner {
        return when (config.runtime) {
            ModelConfig.RUNTIME_LLAMACPP -> LlamaCppTextGenerationAdapter(
                context ?: error("llama.cpp adapter requires an Android application context.")
            )
            // Keep mock explicit so intentional mock catalog entries run without a missing-adapter warning.
            ModelConfig.RUNTIME_MOCK -> MockRuntimeRunner()
            else -> MockRuntimeRunner(
                warning = "No adapter is registered for runtime '${config.runtime}'. Running mock output."
            )
        }
    }
}
