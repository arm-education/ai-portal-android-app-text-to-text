package com.arm.learningpath.texttotext

import java.io.File

interface RuntimeRunner : AutoCloseable {
    fun load(modelDir: File, config: ModelConfig): Long
    fun runTextGeneration(prompt: String): RunResult
    fun runEmbedding(text: String): RunResult
    override fun close()
}
