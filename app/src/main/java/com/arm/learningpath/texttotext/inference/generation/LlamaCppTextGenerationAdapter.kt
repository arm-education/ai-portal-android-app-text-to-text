package com.arm.learningpath.texttotext.inference.generation

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.TextGenerationRunner
import java.io.File

class LlamaCppTextGenerationAdapter : TextGenerationRunner {
    override fun load(modelDir: File, config: ModelConfig): Long {
        val modelFile = File(modelDir, config.modelFile)
        require(modelFile.exists()) {
            "llama.cpp GGUF model file is missing: ${modelFile.absolutePath}"
        }
        if (config.runtimeConfig.isNotBlank()) {
            val runtimeConfig = File(modelDir, config.runtimeConfig)
            require(runtimeConfig.exists()) {
                "llama.cpp runtime config is missing: ${runtimeConfig.absolutePath}"
            }
        }
        error("Apply a validated llama.cpp adapter example before running '${config.id}'.")
    }

    override fun runTextGeneration(prompt: String): RunResult {
        error("Apply a validated llama.cpp adapter example before running text generation. Prompt: $prompt")
    }

    override fun close() {
    }
}
