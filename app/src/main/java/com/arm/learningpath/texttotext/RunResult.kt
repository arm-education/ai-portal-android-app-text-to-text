package com.arm.learningpath.texttotext

data class RunResult(
    val text: String,
    val loadTimeMs: Long,
    val runTimeMs: Long,
)
