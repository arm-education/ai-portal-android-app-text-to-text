package com.arm.learningpath.texttotext.inference


data class RunResult(
    val text: String,
    val loadTimeMs: Long,
    val runTimeMs: Long,
)
