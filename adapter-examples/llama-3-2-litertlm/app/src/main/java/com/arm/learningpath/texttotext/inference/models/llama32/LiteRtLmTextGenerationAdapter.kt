package com.arm.learningpath.texttotext.inference.models.llama32

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.TextGenerationRunner
import java.io.File

class LiteRtLmTextGenerationAdapter : TextGenerationRunner {
    private var config: ModelConfig? = null
    private var engine: Engine? = null
    private var loadTimeMs: Long = 0

    override fun load(modelDir: File, config: ModelConfig): Long {
        val started = System.nanoTime()
        require(config.runtime == ModelConfig.RUNTIME_LITERT_LM) {
            "LiteRtLmTextGenerationAdapter requires runtime 'litert-lm', got '${config.runtime}'."
        }
        require(config.requiresGeneration) {
            "LiteRtLmTextGenerationAdapter supports text generation, got '${config.workload}'."
        }
        require(modelDir.isDirectory) {
            "Model directory is missing: ${modelDir.absolutePath}"
        }

        val bundle = File(modelDir, config.modelFile)
        requireReadableFile(bundle, "LiteRT-LM bundle")
        require(bundle.extension == "litertlm") {
            "LiteRT-LM model file must be a .litertlm bundle: ${bundle.absolutePath}"
        }
        if (config.runtimeConfig.isNotBlank()) {
            requireReadableFile(File(modelDir, config.runtimeConfig), "LiteRT-LM runtime config")
        }
        config.tokenizerFiles.forEach { tokenizerFile ->
            requireReadableFile(File(modelDir, tokenizerFile), "tokenizer file")
        }
        config.externalDataFiles.forEach { externalDataFile ->
            requireReadableFile(File(modelDir, externalDataFile), "external data file")
        }
        require(config.inputTensorNames == listOf("input_ids")) {
            "LiteRT-LM bundle metadata must expose only input_ids; got ${config.inputTensorNames}."
        }
        require(config.outputTensorNames.isEmpty()) {
            "LiteRT-LM streams decoded tokens internally and should not declare tensor outputs; got ${config.outputTensorNames}."
        }

        close()
        val maxTokens = config.maxInputTokens.takeIf { it > 0 }
            ?.let { inputTokens -> inputTokens + config.maxNewTokens.coerceAtLeast(1) }
        val nextEngine = Engine(
            EngineConfig(
                modelPath = bundle.absolutePath,
                backend = Backend.CPU(numOfThreads = LITERT_LM_CPU_THREADS),
                maxNumTokens = maxTokens,
                cacheDir = File(modelDir, CACHE_DIR_NAME).absolutePath,
            )
        )
        nextEngine.initialize()

        engine = nextEngine
        this.config = config
        loadTimeMs = elapsedMs(started)
        return loadTimeMs
    }

    override fun runTextGeneration(prompt: String): RunResult {
        val selected = config ?: error("Call load() before runTextGeneration().")
        val activeEngine = engine ?: error("LiteRT-LM engine is not initialized.")
        val started = System.nanoTime()

        val output = createConversation(activeEngine).use { conversation ->
            conversation.sendMessage(prompt).contents.contents
                .joinToString(separator = "") { content ->
                    when (content) {
                        is com.google.ai.edge.litertlm.Content.Text -> content.text
                        else -> content.toString()
                    }
                }
                .trim()
        }

        return RunResult(output, loadTimeMs, elapsedMs(started))
    }

    override fun close() {
        engine?.close()
        engine = null
        config = null
    }

    private fun createConversation(engine: Engine): Conversation {
        return engine.createConversation(
            ConversationConfig(
                samplerConfig = SamplerConfig(
                    topK = 1,
                    topP = 1.0,
                    temperature = 0.0,
                    seed = 0,
                ),
                channels = emptyList(),
            )
        )
    }

    private fun requireReadableFile(file: File, label: String) {
        require(file.isFile && file.canRead()) {
            "Required $label is missing or unreadable: ${file.absolutePath}"
        }
    }

    private fun elapsedMs(started: Long): Long {
        return (System.nanoTime() - started) / 1_000_000
    }

    private companion object {
        const val LITERT_LM_CPU_THREADS = 4
        const val CACHE_DIR_NAME = "litertlm-cache"
    }
}
