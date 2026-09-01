package com.arm.learningpath.texttotext.inference

interface TextGenerationRunner : RuntimeRunner {
    fun runTextGeneration(prompt: String): RunResult
}
