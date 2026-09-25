package com.arm.learningpath.texttotext.inference.models.qwen

import android.content.Context
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.TextGenerationRunner
import java.io.File
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking

class LlamaCppTextGenerationAdapter(
    context: Context,
) : TextGenerationRunner {
    private val appContext = context.applicationContext
    private var config: ModelConfig? = null
    private var engine: InferenceEngine? = null
    private var loadTimeMs: Long = 0

    override fun load(modelDir: File, config: ModelConfig): Long {
        val started = System.nanoTime()
        require(config.requiresGeneration) {
            "llama.cpp adapter supports text generation only."
        }
        require(config.runtime == ModelConfig.RUNTIME_LLAMACPP) {
            "llama.cpp adapter expected runtime '${ModelConfig.RUNTIME_LLAMACPP}', got '${config.runtime}'."
        }
        validateModelPackage(modelDir, config)

        close()
        val activeEngine = AiChat.getInferenceEngine(appContext)
        val modelFile = File(modelDir, config.modelFile)
        runBlocking {
            activeEngine.loadModel(modelFile.absolutePath)
        }

        engine = activeEngine
        this.config = config
        loadTimeMs = elapsedMs(started)
        return loadTimeMs
    }

    override fun runTextGeneration(prompt: String): RunResult {
        val selected = config ?: error("Call load() before runTextGeneration().")
        val activeEngine = engine ?: error("llama.cpp model is not loaded.")
        val started = System.nanoTime()
        val output = StringBuilder()
        val maxNewTokens = selected.maxNewTokens.coerceAtLeast(1)

        runBlocking {
            activeEngine.sendUserPrompt(prompt, maxNewTokens).collect { token ->
                output.append(token)
            }
        }

        return RunResult(
            text = cleanAssistantCompletion(output.toString(), prompt),
            loadTimeMs = loadTimeMs,
            runTimeMs = elapsedMs(started),
        )
    }

    override fun close() {
        runCatching { engine?.cleanUp() }
        engine = null
        config = null
    }

    private fun validateModelPackage(modelDir: File, config: ModelConfig) {
        require(modelDir.exists() && modelDir.isDirectory) {
            "Model directory is missing: ${modelDir.absolutePath}"
        }
        require(config.modelFile.endsWith(".gguf")) {
            "llama.cpp validated example expects a GGUF model file."
        }
        requireRequiredFile(modelDir, config.modelFile, "GGUF model file")
        if (config.runtimeConfig.isNotBlank()) {
            requireRequiredFile(modelDir, config.runtimeConfig, "Runtime config")
        }
    }

    private fun requireRequiredFile(modelDir: File, relativePath: String, label: String) {
        require(relativePath.isNotBlank()) {
            "$label path is blank."
        }
        require(!relativePath.contains("..") && !File(relativePath).isAbsolute) {
            "$label must be relative to the app-local model directory: $relativePath"
        }
        val basePath = modelDir.canonicalFile.toPath()
        val file = File(modelDir, relativePath).canonicalFile
        require(file.toPath().startsWith(basePath)) {
            "$label must stay below ${modelDir.canonicalPath}: $relativePath"
        }
        require(file.isFile) {
            "$label is missing: ${file.absolutePath}"
        }
    }

    private fun cleanAssistantCompletion(decoded: String, originalPrompt: String): String {
        var cleaned = decoded
            .removePrefix(originalPrompt)
            .replace("<|im_start|>assistant", "")
            .replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .trim()

        val stopMarkers = listOf("<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant", "User:", "System:")
        val markerIndex = stopMarkers
            .map { cleaned.indexOf(it) }
            .filter { it >= 0 }
            .minOrNull()
        if (markerIndex != null) {
            cleaned = cleaned.substring(0, markerIndex).trimEnd()
        }

        return cleaned.trim()
    }

    private fun elapsedMs(started: Long): Long {
        return (System.nanoTime() - started) / 1_000_000
    }
}
