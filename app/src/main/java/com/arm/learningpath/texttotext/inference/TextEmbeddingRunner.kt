package com.arm.learningpath.texttotext.inference

interface TextEmbeddingRunner : RuntimeRunner {
    fun runEmbedding(text: String): RunResult
}
