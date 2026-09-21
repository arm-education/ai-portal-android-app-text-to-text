package com.arm.learningpath.texttotext.inference.models.llama32

import ai.onnxruntime.genai.Generator
import ai.onnxruntime.genai.GeneratorParams
import ai.onnxruntime.genai.GenAI
import ai.onnxruntime.genai.Model
import ai.onnxruntime.genai.Tokenizer
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.TextGenerationRunner
import java.io.File
import kotlin.math.min
import org.json.JSONArray
import org.json.JSONObject

class OnnxTextGenerationAdapter : TextGenerationRunner {
    private var config: ModelConfig? = null
    private var model: Model? = null
    private var tokenizer: Tokenizer? = null
    private var chatTemplate: String = ""
    private var eosTokenIds: Set<Int> = emptySet()
    private var contextLength: Int = 4096
    private var loadTimeMs: Long = 0

    override fun load(modelDir: File, config: ModelConfig): Long {
        val started = System.nanoTime()
        require(config.requiresGeneration) {
            "ONNX Runtime GenAI adapter supports text generation only."
        }

        validateModelBundle(modelDir, config)
        val genaiConfig = JSONObject(resolveModelFile(modelDir, config.runtimeConfig).readText())
        validateGenAiConfig(genaiConfig, config)
        contextLength = genaiConfig.getJSONObject("model").optInt(
            "context_length",
            if (config.maxInputTokens > 0) config.maxInputTokens else 4096,
        )
        chatTemplate = resolveModelFile(modelDir, config.chatTemplate).readText()

        close()
        GenAI.setTelemetry(false)
        val loadedModel = Model(modelDir.canonicalPath)
        val loadedTokenizer = Tokenizer(loadedModel)
        model = loadedModel
        tokenizer = loadedTokenizer
        eosTokenIds = loadedTokenizer.getEosTokenIds().toSet()
        if (eosTokenIds.isEmpty()) {
            eosTokenIds = parseEosTokenIds(genaiConfig)
        }

        this.config = config
        loadTimeMs = elapsedMs(started)
        return loadTimeMs
    }

    override fun runTextGeneration(prompt: String): RunResult {
        val selected = config ?: error("Call load() before runTextGeneration().")
        val activeModel = model ?: error("ONNX Runtime GenAI model is not loaded.")
        val activeTokenizer = tokenizer ?: error("ONNX Runtime GenAI tokenizer is not loaded.")
        val started = System.nanoTime()
        val renderedPrompt = renderPrompt(activeTokenizer, prompt)
        val promptIds = activeTokenizer.encode(renderedPrompt).use { sequences ->
            sequences.getSequence(0)
        }
        require(promptIds.isNotEmpty()) {
            "Prompt produced no tokens."
        }
        require(promptIds.size < contextLength) {
            "Prompt has ${promptIds.size} tokens, which reaches the model context length of $contextLength."
        }

        val requestedNewTokens = selected.maxNewTokens.coerceAtLeast(1)
        val maxLength = min(contextLength, promptIds.size + requestedNewTokens)
        val completionText = GeneratorParams(activeModel).use { params ->
            params.setSearchOption("max_length", maxLength.toDouble())
            params.setSearchOption("do_sample", false)
            params.setSearchOption("temperature", 0.0)
            params.setSearchOption("top_p", 1.0)
            params.setSearchOption("early_stopping", true)

            Generator(activeModel, params).use { generator ->
                generator.appendTokens(promptIds)
                while (!generator.isDone()) {
                    generator.generateNextToken()
                }
                val sequence = generator.getSequence(0)
                val completionIds = sequence
                    .drop(promptIds.size)
                    .takeUntilStopToken(eosTokenIds)
                    .toIntArray()
                activeTokenizer.decode(completionIds)
            }
        }

        return RunResult(cleanAssistantCompletion(completionText, renderedPrompt, prompt), loadTimeMs, elapsedMs(started))
    }

    override fun close() {
        tokenizer?.close()
        tokenizer = null
        model?.close()
        model = null
    }

    private fun elapsedMs(started: Long): Long {
        return (System.nanoTime() - started) / 1_000_000
    }

    private fun validateModelBundle(modelDir: File, config: ModelConfig) {
        val expectedDir = File(modelDir.parentFile ?: File("."), config.id).canonicalFile
        val actualDir = modelDir.canonicalFile
        require(actualDir == expectedDir) {
            "Model files must be loaded from filesDir/models/${config.id}; got ${actualDir.absolutePath}."
        }
        require(actualDir.name == config.id && actualDir.parentFile?.name == "models") {
            "Model directory must end with filesDir/models/${config.id}; got ${actualDir.absolutePath}."
        }
        require(config.artifactPath == config.id) {
            "Catalog artifactPath '${config.artifactPath}' must match model id '${config.id}'."
        }
        require(actualDir.isDirectory) {
            "Model directory is missing: ${actualDir.absolutePath}"
        }

        buildList {
            add(config.modelFile)
            addAll(config.externalDataFiles)
            add(config.runtimeConfig)
            addAll(config.tokenizerFiles)
            add(config.chatTemplate)
        }
            .filter { it.isNotBlank() }
            .distinct()
            .forEach { relativePath ->
                val file = resolveModelFile(actualDir, relativePath)
                require(file.isFile) {
                    "Required ONNX Runtime GenAI file is missing: ${file.absolutePath}"
                }
            }
    }

    private fun resolveModelFile(modelDir: File, relativePath: String): File {
        require(relativePath.isNotBlank()) {
            "Catalog contains a blank model-relative path."
        }
        require(!relativePath.contains("..") && !File(relativePath).isAbsolute) {
            "Catalog contains an unsafe model-relative path: $relativePath"
        }
        val basePath = modelDir.canonicalFile.toPath()
        val file = File(modelDir, relativePath).canonicalFile
        require(file.toPath().startsWith(basePath)) {
            "Catalog path escapes the model directory: $relativePath"
        }
        return file
    }

    private fun validateGenAiConfig(genaiConfig: JSONObject, config: ModelConfig) {
        val modelConfig = genaiConfig.getJSONObject("model")
        require(modelConfig.optString("type") == "llama") {
            "This adapter was validated for ONNX Runtime GenAI Llama bundles only."
        }
        require(modelConfig.optInt("context_length", 0) > 0) {
            "GenAI config is missing a positive context_length."
        }

        val decoder = modelConfig.getJSONObject("decoder")
        require(decoder.optString("filename") == config.modelFile) {
            "GenAI decoder filename '${decoder.optString("filename")}' does not match catalog model file '${config.modelFile}'."
        }

        val inputs = decoder.getJSONObject("inputs")
        require(inputs.optString("input_ids") == "input_ids") {
            "GenAI config does not map semantic input_ids to the expected runtime input."
        }
        require(inputs.has("attention_mask")) {
            "GenAI config is missing the attention_mask mapping required by the generator."
        }

        val outputs = decoder.getJSONObject("outputs")
        require(outputs.optString("logits") == "logits") {
            "GenAI config does not map semantic logits to the expected runtime output."
        }
    }

    private fun parseEosTokenIds(genaiConfig: JSONObject): Set<Int> {
        val eosValue = genaiConfig.getJSONObject("model").opt("eos_token_id")
        return when (eosValue) {
            is JSONArray -> List(eosValue.length()) { index -> eosValue.getInt(index) }.toSet()
            is Number -> setOf(eosValue.toInt())
            else -> emptySet()
        }
    }

    private fun renderPrompt(tokenizer: Tokenizer, prompt: String): String {
        val messages = JSONArray()
            .put(JSONObject().put("role", "user").put("content", prompt))
            .toString()
        return tokenizer.applyChatTemplate(chatTemplate, messages, null, true)
    }

    private fun List<Int>.takeUntilStopToken(stopTokenIds: Set<Int>): List<Int> {
        if (stopTokenIds.isEmpty()) {
            return this
        }
        val stopIndex = indexOfFirst { it in stopTokenIds }
        return if (stopIndex >= 0) take(stopIndex) else this
    }

    private fun cleanAssistantCompletion(decoded: String, renderedPrompt: String, originalPrompt: String): String {
        var cleaned = decoded
            .removePrefix(renderedPrompt)
            .removePrefix(originalPrompt)
            .replace("<|begin_of_text|>", "")
            .replace("<|end_of_text|>", "")
            .replace("<|eot_id|>", "")
            .trim()

        cleaned = cleaned.removePrefix("<|start_header_id|>assistant<|end_header_id|>").trimStart()
        val stopMarkers = listOf(
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
        )
        val markerIndex = stopMarkers
            .map { cleaned.indexOf(it) }
            .filter { it >= 0 }
            .minOrNull()
        if (markerIndex != null) {
            cleaned = cleaned.substring(0, markerIndex).trimEnd()
        }

        return cleaned.trim()
    }
}
