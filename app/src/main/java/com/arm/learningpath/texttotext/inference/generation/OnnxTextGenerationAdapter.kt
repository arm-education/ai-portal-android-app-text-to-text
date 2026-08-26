package com.arm.learningpath.texttotext.inference.generation

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.RuntimeRunner

import java.io.File

class OnnxTextGenerationAdapter : RuntimeRunner {
    private var config: ModelConfig? = null
    private var loadTimeMs: Long = 0

    override fun load(modelDir: File, config: ModelConfig): Long {
        val started = System.nanoTime()
        val modelFile = File(modelDir, config.modelFile)
        require(modelFile.exists()) {
            "ONNX model file is missing: ${modelFile.absolutePath}"
        }

        this.config = config

        // TODO: Add ONNX Runtime or ONNX Runtime GenAI Android dependency.
        // TODO: Load the model directory or model file according to the selected ONNX runtime API.
        // TODO: Map semantic inputs to runtime tensors by meaningful name when available.
        // TODO: If exported tensor names are internal, fall back to metadata, order, shape, and dtype.
        // TODO: Initialize tokenizer and generation settings from the model directory.
        loadTimeMs = elapsedMs(started)
        return loadTimeMs
    }

    override fun runTextGeneration(prompt: String): RunResult {
        val selected = config ?: error("Call load() before runTextGeneration().")
        val started = System.nanoTime()

        // TODO: Tokenize prompt, run the ONNX generation API or autoregressive loop, and decode output.
        val output = """
            ONNX adapter is ready for ${selected.displayName}, but real inference is not implemented yet.
            Prompt: $prompt
        """.trimIndent()

        return RunResult(output, loadTimeMs, elapsedMs(started))
    }

    override fun runEmbedding(text: String): RunResult {
        error("OnnxTextGenerationAdapter currently supports text generation only.")
    }

    override fun close() {
        // TODO: Release ONNX runtime session resources.
    }

    private fun elapsedMs(started: Long): Long {
        return (System.nanoTime() - started) / 1_000_000
    }
}
