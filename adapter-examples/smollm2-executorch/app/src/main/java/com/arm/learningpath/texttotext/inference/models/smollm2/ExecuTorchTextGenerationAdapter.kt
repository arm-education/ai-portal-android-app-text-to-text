package com.arm.learningpath.texttotext.inference.models.smollm2

import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.TextGenerationRunner
import java.io.File
import kotlin.math.min
import kotlin.system.measureTimeMillis

class ExecuTorchTextGenerationAdapter : TextGenerationRunner {
    private var module: Module? = null
    private var tokenizer: ByteLevelBpeTokenizer? = null
    private var loadTimeMs: Long = 0L
    private var eosTokenId: Long = 2L
    private var maxInputTokens: Int = 2047
    private var maxNewTokens: Int = 64

    override fun load(modelDir: File, config: ModelConfig): Long {
        close()
        val elapsed = measureTimeMillis {
            validateModelFiles(modelDir, config)
            val runtimeCacheDir = File(modelDir, "executorch-cache")
            require(runtimeCacheDir.isDirectory || runtimeCacheDir.mkdirs()) {
                "Could not create ExecuTorch cache directory: ${runtimeCacheDir.absolutePath}"
            }

            val nextTokenizer = ByteLevelBpeTokenizer.fromFile(File(modelDir, "tokenizer/tokenizer.json"))
            val nextModule = Module.load(File(modelDir, config.modelFile).absolutePath)

            maxInputTokens = config.maxInputTokens.takeIf { it > 0 } ?: 2047
            maxNewTokens = config.maxNewTokens.takeIf { it > 0 } ?: 64
            eosTokenId = nextTokenizer.idForToken("<|im_end|>") ?: nextTokenizer.idForToken("<|endoftext|>") ?: 2L

            module = nextModule
            tokenizer = nextTokenizer
            validateForwardContract()
        }
        loadTimeMs = elapsed
        return elapsed
    }

    override fun runTextGeneration(prompt: String): RunResult {
        val activeTokenizer = requireNotNull(tokenizer) { "ExecuTorch model is not loaded." }
        val promptIds = activeTokenizer.encode(applyChatTemplate(prompt))
            .takeLast(maxInputTokens)
            .map { it.toLong() }
            .toMutableList()
        require(promptIds.isNotEmpty()) {
            "Prompt produced no tokens."
        }

        val generatedIds = mutableListOf<Long>()
        val runMs = measureTimeMillis {
            var inputIds = promptIds
            var cacheStart = 0L
            repeat(maxNewTokens) {
                val logits = forward(inputIds, cacheStart)
                val nextToken = argmaxLastToken(logits, inputIds.size)
                if (nextToken == eosTokenId) {
                    return@measureTimeMillis
                }
                generatedIds.add(nextToken)
                inputIds = mutableListOf(nextToken)
                cacheStart = (promptIds.size + generatedIds.size - 1).toLong()
                if (cacheStart >= 2047L) {
                    return@measureTimeMillis
                }
            }
        }

        val completion = sanitizeCompletion(activeTokenizer.decode(generatedIds))
        return RunResult(text = completion, loadTimeMs = loadTimeMs, runTimeMs = runMs)
    }

    override fun close() {
        module?.destroy()
        module = null
        tokenizer = null
    }

    private fun validateModelFiles(modelDir: File, config: ModelConfig) {
        require(config.runtime == ModelConfig.RUNTIME_EXECUTORCH) {
            "ExecuTorchTextGenerationAdapter cannot run runtime '${config.runtime}'."
        }
        require(config.workload == ModelConfig.WORKLOAD_TEXT_GENERATION) {
            "ExecuTorchTextGenerationAdapter only supports text generation."
        }
        if (config.modelFile.isNotBlank()) {
            requireExistingFile(modelDir, config.modelFile, "ExecuTorch model file")
        }
        if (config.runtimeConfig.isNotBlank()) {
            requireExistingFile(modelDir, config.runtimeConfig, "Runtime config file")
        }
        config.tokenizerFiles.forEach { requireExistingFile(modelDir, it, "Tokenizer file") }
        if (config.chatTemplate.isNotBlank()) {
            requireExistingFile(modelDir, config.chatTemplate, "Chat template file")
        }
        config.externalDataFiles.forEach { requireExistingFile(modelDir, it, "External data file") }
    }

    private fun requireExistingFile(modelDir: File, relativePath: String, label: String) {
        require(relativePath.isNotBlank()) {
            "$label path is blank."
        }
        require(!File(relativePath).isAbsolute) {
            "$label must be relative to the app-local model directory: $relativePath"
        }
        val canonicalModelDir = modelDir.canonicalFile
        val file = File(modelDir, relativePath).canonicalFile
        require(file.path == canonicalModelDir.path || file.path.startsWith(canonicalModelDir.path + File.separator)) {
            "$label must stay below ${canonicalModelDir.absolutePath}: $relativePath"
        }
        require(file.isFile) {
            "$label is missing: ${file.absolutePath}"
        }
    }

    private fun validateForwardContract() {
        val probeIds = listOf(eosTokenId)
        val logits = forward(probeIds, 0L)
        val expected = probeIds.size * VOCAB_SIZE
        require(logits.size >= expected) {
            "ExecuTorch output is too small for [1, T, $VOCAB_SIZE] logits: ${logits.size} floats."
        }
    }

    private fun forward(tokenIds: List<Long>, cacheStart: Long): FloatArray {
        require(tokenIds.isNotEmpty()) {
            "ExecuTorch input_ids cannot be empty."
        }
        require(tokenIds.size <= maxInputTokens) {
            "Input has ${tokenIds.size} tokens, but this model accepts at most $maxInputTokens."
        }
        val positions = LongArray(tokenIds.size) { index -> cacheStart + index }
        val inputTensor = Tensor.fromBlob(tokenIds.toLongArray(), longArrayOf(1L, tokenIds.size.toLong()))
        val positionTensor = Tensor.fromBlob(positions, longArrayOf(tokenIds.size.toLong()))
        val outputs = requireNotNull(module) { "ExecuTorch model is not loaded." }.forward(
            EValue.from(inputTensor),
            EValue.from(positionTensor),
        )
        require(outputs.isNotEmpty()) {
            "ExecuTorch returned no outputs."
        }
        return outputs[0].toTensor().dataAsFloatArray
    }

    private fun argmaxLastToken(logits: FloatArray, tokenCount: Int): Long {
        val start = (tokenCount - 1) * VOCAB_SIZE
        require(logits.size >= start + VOCAB_SIZE) {
            "ExecuTorch logits do not contain the last token distribution."
        }
        var bestIndex = 0
        var bestValue = logits[start]
        for (index in 1 until VOCAB_SIZE) {
            val value = logits[start + index]
            if (value > bestValue) {
                bestValue = value
                bestIndex = index
            }
        }
        return bestIndex.toLong()
    }

    private fun applyChatTemplate(prompt: String): String {
        return "<|im_start|>system\n" +
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n" +
            "<|im_start|>user\n" +
            prompt.trim() +
            "<|im_end|>\n" +
            "<|im_start|>assistant\n"
    }

    private fun sanitizeCompletion(text: String): String {
        val markers = listOf("<|im_end|>", "<|im_start|>", "<|endoftext|>", "\nuser\n", "\nsystem\n")
        var end = text.length
        markers.forEach { marker ->
            val index = text.indexOf(marker)
            if (index >= 0) {
                end = min(end, index)
            }
        }
        return text.substring(0, end)
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .trim()
    }

    private class ByteLevelBpeTokenizer(
        private val vocab: Map<String, Int>,
        private val idToToken: Map<Int, String>,
        private val mergeRanks: Map<String, Int>,
        private val specialTokens: Map<String, Int>,
    ) {
        private val byteEncoder = buildByteEncoder()
        private val byteDecoder = byteEncoder.entries.associate { (byteValue, charValue) -> charValue to byteValue.toByte() }
        private val bpeCache = mutableMapOf<String, List<String>>()
        private val specialPattern = specialTokens.keys.sortedByDescending { it.length }
        private val tokenPattern = Regex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

        fun idForToken(token: String): Long? = vocab[token]?.toLong()

        fun encode(text: String): List<Int> {
            val ids = mutableListOf<Int>()
            var index = 0
            while (index < text.length) {
                val special = specialPattern.firstOrNull { text.startsWith(it, index) }
                if (special != null) {
                    ids.add(requireNotNull(specialTokens[special]))
                    index += special.length
                    continue
                }

                val nextSpecialIndex = specialPattern
                    .map { text.indexOf(it, startIndex = index) }
                    .filter { it >= 0 }
                    .minOrNull() ?: text.length
                val chunk = text.substring(index, nextSpecialIndex)
                tokenPattern.findAll(chunk).forEach { match ->
                    val encoded = match.value.toByteArray(Charsets.UTF_8).joinToString("") { byte ->
                        byteEncoder[byte.toInt() and 0xff].toString()
                    }
                    bpe(encoded).forEach { token ->
                        ids.add(vocab[token] ?: error("Tokenizer vocabulary is missing token '$token'."))
                    }
                }
                index = nextSpecialIndex
            }
            return ids
        }

        fun decode(ids: List<Long>): String {
            val bytes = mutableListOf<Byte>()
            ids.forEach { id ->
                val token = idToToken[id.toInt()] ?: return@forEach
                if (specialTokens.containsKey(token)) {
                    return@forEach
                }
                token.forEach { char ->
                    byteDecoder[char]?.let { bytes.add(it) }
                }
            }
            return bytes.toByteArray().toString(Charsets.UTF_8)
        }

        private fun bpe(token: String): List<String> {
            bpeCache[token]?.let { return it }
            var parts = token.map { it.toString() }
            if (parts.size <= 1) {
                return parts
            }

            while (true) {
                var bestRank = Int.MAX_VALUE
                var bestIndex = -1
                for (index in 0 until parts.lastIndex) {
                    val rank = mergeRanks[parts[index] + "\u0000" + parts[index + 1]] ?: continue
                    if (rank < bestRank) {
                        bestRank = rank
                        bestIndex = index
                    }
                }
                if (bestIndex < 0) {
                    break
                }
                val merged = parts[bestIndex] + parts[bestIndex + 1]
                parts = parts.take(bestIndex) + merged + parts.drop(bestIndex + 2)
                if (parts.size == 1) {
                    break
                }
            }
            bpeCache[token] = parts
            return parts
        }

        companion object {
            fun fromFile(file: File): ByteLevelBpeTokenizer {
                val model = JSONObject(file.readText(Charsets.UTF_8)).getJSONObject("model")
                val vocabObject = model.getJSONObject("vocab")
                val vocab = mutableMapOf<String, Int>()
                vocabObject.keys().forEach { token ->
                    vocab[token] = vocabObject.getInt(token)
                }

                val addedTokens = JSONObject(file.readText(Charsets.UTF_8)).getJSONArray("added_tokens")
                val specials = mutableMapOf<String, Int>()
                for (index in 0 until addedTokens.length()) {
                    val item = addedTokens.getJSONObject(index)
                    if (item.optBoolean("special")) {
                        specials[item.getString("content")] = item.getInt("id")
                    }
                }

                val merges = model.getJSONArray("merges")
                val ranks = mutableMapOf<String, Int>()
                for (index in 0 until merges.length()) {
                    val pair = merges.getJSONArray(index)
                    ranks[pair.getString(0) + "\u0000" + pair.getString(1)] = index
                }

                return ByteLevelBpeTokenizer(
                    vocab = vocab,
                    idToToken = vocab.entries.associate { (token, id) -> id to token },
                    mergeRanks = ranks,
                    specialTokens = specials,
                )
            }

            private fun buildByteEncoder(): Map<Int, Char> {
                val bytes = mutableListOf<Int>()
                bytes.addAll(33..126)
                bytes.addAll(161..172)
                bytes.addAll(174..255)
                val chars = bytes.toMutableList()
                var next = 0
                for (byteValue in 0..255) {
                    if (!bytes.contains(byteValue)) {
                        bytes.add(byteValue)
                        chars.add(256 + next)
                        next += 1
                    }
                }
                return bytes.zip(chars).associate { (byteValue, charValue) -> byteValue to charValue.toChar() }
            }
        }
    }

    private companion object {
        const val VOCAB_SIZE = 49152
    }
}
