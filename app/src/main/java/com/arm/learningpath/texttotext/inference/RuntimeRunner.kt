package com.arm.learningpath.texttotext.inference

import com.arm.learningpath.texttotext.catalog.ModelConfig

import java.io.File

interface RuntimeRunner : AutoCloseable {
    fun load(modelDir: File, config: ModelConfig): Long
    override fun close()
}

interface TextGenerationRunner : RuntimeRunner {
    fun runTextGeneration(prompt: String): RunResult
}

interface TextEmbeddingRunner : RuntimeRunner {
    fun runEmbedding(text: String): RunResult
}
