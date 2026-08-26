package com.arm.learningpath.texttotext.inference.generation

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.RuntimeRunner

import java.io.File

class ExecuTorchTextGenerationAdapter : RuntimeRunner {
    override fun load(modelDir: File, config: ModelConfig): Long {
        validateModelFiles(modelDir, config)
        // TODO: Map semantic inputs to runtime tensors by meaningful name when available.
        // TODO: If exported tensor names are internal, fall back to metadata, order, shape, and dtype.
        error("Replace this stub with ExecuTorch text-generation loading code for '${config.id}'.")
    }

    override fun runTextGeneration(prompt: String): RunResult {
        error("Replace this stub with ExecuTorch generation code. Prompt: $prompt")
    }

    override fun runEmbedding(text: String): RunResult {
        error("ExecuTorchTextGenerationAdapter supports text generation, not embeddings.")
    }

    override fun close() {
    }

    private fun validateModelFiles(modelDir: File, config: ModelConfig) {
        if (config.modelFile.isNotBlank()) {
            requireExistingFile(modelDir, config.modelFile, "ExecuTorch model file")
        }
        config.tokenizerFiles.forEach { requireExistingFile(modelDir, it, "Tokenizer file") }
        if (config.chatTemplate.isNotBlank()) {
            requireExistingFile(modelDir, config.chatTemplate, "Chat template file")
        }
        config.externalDataFiles.forEach { requireExistingFile(modelDir, it, "External data file") }
    }

    private fun requireExistingFile(modelDir: File, relativePath: String, label: String) {
        require(relativePath.isNotBlank()) {
            "$label path is blank."
        }
        require(!File(relativePath).isAbsolute) {
            "$label must be relative to the app-local model directory: $relativePath"
        }
        val canonicalModelDir = modelDir.canonicalFile
        val file = File(modelDir, relativePath).canonicalFile
        require(file.path == canonicalModelDir.path || file.path.startsWith(canonicalModelDir.path + File.separator)) {
            "$label must stay below ${canonicalModelDir.absolutePath}: $relativePath"
        }
        require(file.isFile) {
            "$label is missing: ${file.absolutePath}"
        }
    }
}
