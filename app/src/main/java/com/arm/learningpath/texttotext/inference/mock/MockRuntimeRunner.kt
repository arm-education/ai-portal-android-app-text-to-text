package com.arm.learningpath.texttotext.inference.mock

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.TextEmbeddingRunner
import com.arm.learningpath.texttotext.inference.TextGenerationRunner

import java.io.File

class MockRuntimeRunner(
    private val warning: String? = null,
) : TextGenerationRunner, TextEmbeddingRunner {
    private var config: ModelConfig? = null
    private var loadTimeMs: Long = 0

    override fun load(modelDir: File, config: ModelConfig): Long {
        val started = System.nanoTime()
        this.config = config
        loadTimeMs = elapsedMs(started)
        return loadTimeMs
    }

    override fun runTextGeneration(prompt: String): RunResult {
        val started = System.nanoTime()
        val selected = config ?: error("Call load() before runTextGeneration().")
        val text = buildString {
            warning?.let {
                appendLine(it)
                appendLine()
            }
            appendLine("Mock text-generation result for '${selected.displayName}'.")
            appendLine("Prompt: $prompt")
            append("Replace ${selected.runtime} adapter code to run the real model.")
        }
        return RunResult(text, loadTimeMs, elapsedMs(started))
    }

    override fun runEmbedding(text: String): RunResult {
        val started = System.nanoTime()
        val selected = config ?: error("Call load() before runEmbedding().")
        val dimensions = if (selected.embeddingDimensions > 0) selected.embeddingDimensions else 8
        val preview = List(minOf(dimensions, 8)) { index ->
            val value = ((text.hashCode() + index * 31) % 1000) / 1000.0
            "%.3f".format(value)
        }.joinToString(prefix = "[", postfix = if (dimensions > 8) ", ...]" else "]")

        val result = buildString {
            warning?.let {
                appendLine(it)
                appendLine()
            }
            appendLine("Mock embedding result for '${selected.displayName}'.")
            appendLine("Dimensions: $dimensions")
            append("Preview: $preview")
        }
        return RunResult(result, loadTimeMs, elapsedMs(started))
    }

    override fun close() = Unit

    private fun elapsedMs(started: Long): Long {
        return (System.nanoTime() - started) / 1_000_000
    }
}
