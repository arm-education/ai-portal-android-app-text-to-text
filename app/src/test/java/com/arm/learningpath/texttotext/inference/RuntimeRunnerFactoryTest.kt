package com.arm.learningpath.texttotext.inference

import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.embedding.LiteRtEmbeddingAdapter
import com.arm.learningpath.texttotext.inference.generation.ExecuTorchTextGenerationAdapter
import com.arm.learningpath.texttotext.inference.generation.LiteRtLmTextGenerationAdapter
import com.arm.learningpath.texttotext.inference.generation.OnnxTextGenerationAdapter
import com.arm.learningpath.texttotext.inference.mock.MockRuntimeRunner
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeRunnerFactoryTest {
    @Test
    fun mapsGenerationRuntimesToGenerationRunners() {
        assertTrue(runnerFor(ModelConfig.RUNTIME_EXECUTORCH) is ExecuTorchTextGenerationAdapter)
        assertTrue(runnerFor(ModelConfig.RUNTIME_LITERT_LM) is LiteRtLmTextGenerationAdapter)
        assertTrue(runnerFor(ModelConfig.RUNTIME_ONNX) is OnnxTextGenerationAdapter)
        assertTrue(runnerFor(ModelConfig.RUNTIME_ONNX_RUNTIME) is OnnxTextGenerationAdapter)
        assertTrue(runnerFor(ModelConfig.RUNTIME_ONNX_RUNTIME_GENAI) is OnnxTextGenerationAdapter)
    }

    @Test
    fun mapsEmbeddingRuntimesToEmbeddingRunner() {
        assertTrue(runnerFor(ModelConfig.RUNTIME_LITERT) is LiteRtEmbeddingAdapter)
        assertTrue(runnerFor(ModelConfig.RUNTIME_TFLITE) is LiteRtEmbeddingAdapter)
    }

    @Test
    fun mapsMockAndUnknownRuntimesToMockRunner() {
        assertTrue(runnerFor(ModelConfig.RUNTIME_MOCK) is MockRuntimeRunner)
        assertTrue(runnerFor("unknown-runtime") is MockRuntimeRunner)
    }

    private fun runnerFor(runtime: String): RuntimeRunner {
        return RuntimeRunnerFactory.create(config(runtime))
    }

    private fun config(runtime: String): ModelConfig {
        return ModelConfig(
            id = "sample",
            displayName = "Sample",
            workload = ModelConfig.WORKLOAD_TEXT_GENERATION,
            runtime = runtime,
            artifactType = "directory",
            artifactPath = "sample",
            modelFile = "",
            externalDataFiles = emptyList(),
            runtimeConfig = "",
            tokenizerFiles = emptyList(),
            chatTemplate = "",
            maxInputTokens = 0,
            maxNewTokens = 0,
            embeddingDimensions = 0,
            normalizeEmbeddings = false,
            inputTensorNames = emptyList(),
            outputTensorNames = emptyList(),
        )
    }
}
