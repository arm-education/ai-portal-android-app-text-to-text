package com.arm.learningpath.texttotext.catalog

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ModelConfigTest {
    @Test
    fun parsesModelConfigFromJson() {
        val config = JSONObject(
            """
            {
              "id": "sample",
              "displayName": "Sample",
              "workload": "text-generation",
              "runtime": "executorch",
              "artifactType": "file",
              "artifactPath": "sample",
              "modelFile": "model.pte",
              "externalDataFiles": ["weights.bin"],
              "runtimeConfig": "config.yaml",
              "tokenizerFiles": ["tokenizer.json"],
              "chatTemplate": "chat_template.jinja",
              "maxInputTokens": 128,
              "maxNewTokens": 32,
              "embeddingDimensions": 0,
              "normalizeEmbeddings": false,
              "inputTensorNames": ["input_ids"],
              "outputTensorNames": ["logits"]
            }
            """.trimIndent()
        ).toModelConfig()

        assertEquals("sample", config.id)
        assertEquals("Sample", config.displayName)
        assertEquals(ModelConfig.WORKLOAD_TEXT_GENERATION, config.workload)
        assertEquals(ModelConfig.RUNTIME_EXECUTORCH, config.runtime)
        assertEquals(listOf("weights.bin"), config.externalDataFiles)
        assertEquals(listOf("tokenizer.json"), config.tokenizerFiles)
        assertEquals(listOf("input_ids"), config.inputTensorNames)
        assertEquals(listOf("logits"), config.outputTensorNames)
        assertTrue(config.requiresGeneration)
        assertFalse(config.requiresEmbedding)
    }

    @Test
    fun parsesDefaultCatalogAsset() {
        val catalogFile = File("src/main/assets/model_catalog.json")
        val entries = JSONArray(catalogFile.readText())
        val configs = List(entries.length()) { index ->
            entries.getJSONObject(index).toModelConfig()
        }

        assertEquals(2, configs.size)
        assertTrue(configs.any { it.requiresGeneration })
        assertTrue(configs.any { it.requiresEmbedding })
        assertTrue(configs.all { it.runtime == ModelConfig.RUNTIME_MOCK })
    }
}
