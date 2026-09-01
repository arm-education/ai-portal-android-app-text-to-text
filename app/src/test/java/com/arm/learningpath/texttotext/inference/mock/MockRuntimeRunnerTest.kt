package com.arm.learningpath.texttotext.inference.mock

import com.arm.learningpath.texttotext.catalog.ModelConfig
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class MockRuntimeRunnerTest {
    @Test
    fun returnsTextGenerationResultAfterLoad() {
        val runner = MockRuntimeRunner()
        val config = config(
            workload = ModelConfig.WORKLOAD_TEXT_GENERATION,
            embeddingDimensions = 0,
        )

        runner.load(File("."), config)
        val result = runner.runTextGeneration("Explain on-device AI.")

        assertTrue(result.text.contains("Mock text-generation result"))
        assertTrue(result.text.contains("Explain on-device AI."))
        assertTrue(result.loadTimeMs >= 0)
        assertTrue(result.runTimeMs >= 0)
    }

    @Test
    fun returnsEmbeddingResultWithConfiguredDimensions() {
        val runner = MockRuntimeRunner()
        val config = config(
            workload = ModelConfig.WORKLOAD_TEXT_EMBEDDING,
            embeddingDimensions = 384,
        )

        runner.load(File("."), config)
        val result = runner.runEmbedding("Search text")

        assertTrue(result.text.contains("Mock embedding result"))
        assertTrue(result.text.contains("Dimensions: 384"))
        assertTrue(result.loadTimeMs >= 0)
        assertTrue(result.runTimeMs >= 0)
    }

    private fun config(workload: String, embeddingDimensions: Int): ModelConfig {
        return ModelConfig(
            id = "sample",
            displayName = "Sample",
            workload = workload,
            runtime = ModelConfig.RUNTIME_MOCK,
            artifactType = "directory",
            artifactPath = "sample",
            modelFile = "",
            externalDataFiles = emptyList(),
            runtimeConfig = "",
            tokenizerFiles = emptyList(),
            chatTemplate = "",
            maxInputTokens = 0,
            maxNewTokens = 64,
            embeddingDimensions = embeddingDimensions,
            normalizeEmbeddings = false,
            inputTensorNames = emptyList(),
            outputTensorNames = emptyList(),
        )
    }
}
