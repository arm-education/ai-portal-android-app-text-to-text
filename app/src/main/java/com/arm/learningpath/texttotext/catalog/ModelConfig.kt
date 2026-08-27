package com.arm.learningpath.texttotext.catalog


import org.json.JSONArray
import org.json.JSONObject

data class ModelConfig(
    val id: String,
    val displayName: String,
    val workload: String,
    val runtime: String,
    val artifactType: String,
    val artifactPath: String,
    val modelFile: String,
    val externalDataFiles: List<String>,
    val runtimeConfig: String,
    val tokenizerFiles: List<String>,
    val chatTemplate: String,
    val maxInputTokens: Int,
    val maxNewTokens: Int,
    val embeddingDimensions: Int,
    val normalizeEmbeddings: Boolean,
    val inputTensorNames: List<String>,
    val outputTensorNames: List<String>,
) {
    val requiresGeneration: Boolean
        get() = workload == WORKLOAD_TEXT_GENERATION

    val requiresEmbedding: Boolean
        get() = workload == WORKLOAD_TEXT_EMBEDDING

    companion object {
        const val WORKLOAD_TEXT_GENERATION = "text-generation"
        const val WORKLOAD_TEXT_EMBEDDING = "text-embedding"
        const val RUNTIME_EXECUTORCH = "executorch"
        const val RUNTIME_LITERT_LM = "litert-lm"
        const val RUNTIME_LITERT = "litert"
        const val RUNTIME_TFLITE = "tflite"
        const val RUNTIME_ONNX = "onnx"
        const val RUNTIME_ONNX_RUNTIME = "onnxruntime"
        const val RUNTIME_ONNX_RUNTIME_GENAI = "onnxruntime-genai"
        const val RUNTIME_MOCK = "mock"
    }
}

fun JSONObject.toModelConfig(): ModelConfig {
    return ModelConfig(
        id = getString("id"),
        displayName = getString("displayName"),
        workload = getString("workload"),
        runtime = getString("runtime"),
        artifactType = getString("artifactType"),
        artifactPath = getString("artifactPath"),
        modelFile = getString("modelFile"),
        externalDataFiles = optJSONArray("externalDataFiles").toStringList(),
        runtimeConfig = optString("runtimeConfig"),
        tokenizerFiles = optJSONArray("tokenizerFiles").toStringList(),
        chatTemplate = optString("chatTemplate"),
        maxInputTokens = optInt("maxInputTokens"),
        maxNewTokens = optInt("maxNewTokens"),
        embeddingDimensions = optInt("embeddingDimensions"),
        normalizeEmbeddings = optBoolean("normalizeEmbeddings"),
        inputTensorNames = optJSONArray("inputTensorNames").toStringList(),
        outputTensorNames = optJSONArray("outputTensorNames").toStringList(),
    )
}

private fun JSONArray?.toStringList(): List<String> {
    if (this == null) {
        return emptyList()
    }

    return List(length()) { index -> getString(index) }
}
