package com.arm.learningpath.texttotext.inference.models.bge

import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RunResult
import com.arm.learningpath.texttotext.inference.RuntimeRunner
import java.io.File
import java.text.Normalizer
import java.util.Locale
import kotlin.math.sqrt

class LiteRtEmbeddingAdapter : RuntimeRunner {
    private var config: ModelConfig? = null
    private var interpreter: Interpreter? = null
    private var tokenizer: BertWordPieceTokenizer? = null
    private var inputIdsInputIndex: Int = -1
    private var attentionMaskInputIndex: Int = -1
    private var outputTensorIndex: Int = -1
    private var loadTimeMs: Long = 0

    override fun load(modelDir: File, config: ModelConfig): Long {
        val started = System.nanoTime()
        require(config.requiresEmbedding) {
            "LiteRT embedding adapter cannot run workload '${config.workload}'."
        }
        require(config.runtime == "litert" || config.runtime == "tflite") {
            "LiteRT embedding adapter cannot run runtime '${config.runtime}'."
        }

        val modelFile = requireFileBelow(modelDir, config.modelFile, "LiteRT model")
        val runtimeConfigFile = requireFileBelow(modelDir, config.runtimeConfig, "LiteRT runtime config")
        config.externalDataFiles.forEach { requireFileBelow(modelDir, it, "LiteRT external data") }
        val tokenizerFiles = config.tokenizerFiles.associateWith { fileName ->
            requireFileBelow(modelDir, fileName, "tokenizer")
        }
        val vocabFile = tokenizerFiles["vocab.txt"]
            ?: error("Tokenizer file vocab.txt is required for BERT WordPiece tokenization.")
        val specialTokensFile = tokenizerFiles["special_tokens_map.json"]
            ?: error("Tokenizer file special_tokens_map.json is required.")

        close()
        val nextInterpreter = Interpreter(modelFile)
        validateAndMapTensors(nextInterpreter, config)

        this.interpreter = nextInterpreter
        this.tokenizer = BertWordPieceTokenizer(vocabFile, specialTokensFile, config.maxInputTokens)
        this.config = config

        require(runtimeConfigFile.readText().contains("shape: [1, 128]")) {
            "Runtime config does not match the expected fixed [1, 128] input shape."
        }

        loadTimeMs = elapsedMs(started)
        return loadTimeMs
    }

    override fun runTextGeneration(prompt: String): RunResult {
        error("LiteRtEmbeddingAdapter supports embeddings, not text generation.")
    }

    override fun runEmbedding(text: String): RunResult {
        val selected = config ?: error("Call load() before runEmbedding().")
        val activeInterpreter = interpreter ?: error("LiteRT interpreter is not loaded.")
        val activeTokenizer = tokenizer ?: error("Tokenizer is not loaded.")
        val started = System.nanoTime()

        val encoded = activeTokenizer.encode(text)
        val orderedInputs = arrayOfNulls<Any>(activeInterpreter.inputTensorCount)
        orderedInputs[inputIdsInputIndex] = arrayOf(encoded.inputIds)
        orderedInputs[attentionMaskInputIndex] = arrayOf(encoded.attentionMask)

        val embeddingOutput = arrayOf(FloatArray(selected.embeddingDimensions))
        activeInterpreter.runForMultipleInputsOutputs(
            orderedInputs.requireNoNulls(),
            mapOf(outputTensorIndex to embeddingOutput),
        )

        val embedding = if (selected.normalizeEmbeddings) {
            l2Normalize(embeddingOutput[0])
        } else {
            embeddingOutput[0]
        }
        val norm = vectorNorm(embedding)
        val preview = embedding.take(12).joinToString(
            separator = ", ",
            prefix = "[",
            postfix = if (embedding.size > 12) ", ...]" else "]",
        ) { "%+.6f".format(Locale.US, it) }

        val output = """
            Model: ${selected.displayName}
            Input tokens: ${encoded.tokenCount}/${selected.maxInputTokens}
            Embedding dimensions: ${embedding.size}
            L2 norm: ${"%.6f".format(Locale.US, norm)}
            Embedding preview: $preview
        """.trimIndent()

        return RunResult(output, loadTimeMs, elapsedMs(started))
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        tokenizer = null
        config = null
        inputIdsInputIndex = -1
        attentionMaskInputIndex = -1
        outputTensorIndex = -1
    }

    private fun validateAndMapTensors(interpreter: Interpreter, config: ModelConfig) {
        require(interpreter.inputTensorCount == 2) {
            "Expected exactly 2 LiteRT inputs, found ${interpreter.inputTensorCount}."
        }
        require(interpreter.outputTensorCount == 1) {
            "Expected exactly 1 LiteRT output, found ${interpreter.outputTensorCount}."
        }

        val inputs = List(interpreter.inputTensorCount) { index ->
            TensorInfo(index, interpreter.getInputTensor(index))
        }
        val inputByName = inputs.associateBy { it.name }
        val inputIdsExact = inputByName["input_ids"]
        val attentionMaskExact = inputByName["attention_mask"]
        if (inputIdsExact != null && attentionMaskExact != null) {
            inputIdsInputIndex = inputIdsExact.index
            attentionMaskInputIndex = attentionMaskExact.index
        } else {
            val inputIdsNamed = inputs.firstOrNull { it.name.contains("input_ids", ignoreCase = true) }
            val attentionMaskNamed = inputs.firstOrNull { it.name.contains("attention_mask", ignoreCase = true) }
            if (inputIdsNamed != null && attentionMaskNamed != null) {
                inputIdsInputIndex = inputIdsNamed.index
                attentionMaskInputIndex = attentionMaskNamed.index
            } else {
                inputIdsInputIndex = 0
                attentionMaskInputIndex = 1
            }
        }

        val inputIds = inputs[inputIdsInputIndex]
        val attentionMask = inputs[attentionMaskInputIndex]
        validateInputTensor(inputIds, "input_ids", config.maxInputTokens)
        validateInputTensor(attentionMask, "attention_mask", config.maxInputTokens)

        val output = TensorInfo(0, interpreter.getOutputTensor(0))
        require(output.dataType == DataType.FLOAT32) {
            "Output tensor '${output.name}' must be FLOAT32, found ${output.dataType}."
        }
        require(output.shape.size == 2 && acceptsDim(output.shape[0], 1) && acceptsDim(output.shape[1], config.embeddingDimensions)) {
            "Output tensor '${output.name}' must have shape [1, ${config.embeddingDimensions}], found ${output.shape.contentToString()}."
        }
        outputTensorIndex = 0
    }

    private fun validateInputTensor(tensor: TensorInfo, semanticName: String, maxInputTokens: Int) {
        require(tensor.dataType == DataType.INT64) {
            "Input tensor for $semanticName must be INT64, found ${tensor.dataType} on '${tensor.name}'."
        }
        require(tensor.shape.size == 2 && acceptsDim(tensor.shape[0], 1) && acceptsDim(tensor.shape[1], maxInputTokens)) {
            "Input tensor for $semanticName must have shape [1, $maxInputTokens], found ${tensor.shape.contentToString()} on '${tensor.name}'."
        }
    }

    private fun requireFileBelow(modelDir: File, relativePath: String, label: String): File {
        require(relativePath.isNotBlank()) {
            "$label path is empty."
        }
        require(!File(relativePath).isAbsolute) {
            "$label path must be relative to the app-local model directory."
        }

        val base = modelDir.canonicalFile
        val child = File(base, relativePath).canonicalFile
        val basePath = base.path + File.separator
        require(child.path.startsWith(basePath)) {
            "$label path escapes the app-local model directory: $relativePath"
        }
        require(child.isFile) {
            "$label file is missing: ${child.absolutePath}"
        }
        return child
    }

    private fun acceptsDim(actual: Int, expected: Int): Boolean {
        return actual == expected || actual == -1
    }

    private fun l2Normalize(values: FloatArray): FloatArray {
        val norm = vectorNorm(values).coerceAtLeast(1.0e-12f)
        return FloatArray(values.size) { index -> values[index] / norm }
    }

    private fun vectorNorm(values: FloatArray): Float {
        var sum = 0.0
        values.forEach { value -> sum += value * value }
        return sqrt(sum).toFloat()
    }

    private fun elapsedMs(started: Long): Long {
        return (System.nanoTime() - started) / 1_000_000
    }

    private data class TensorInfo(
        val index: Int,
        val name: String,
        val dataType: DataType,
        val shape: IntArray,
    ) {
        constructor(index: Int, tensor: Tensor) : this(
            index = index,
            name = tensor.name(),
            dataType = tensor.dataType(),
            shape = tensor.shape(),
        )
    }

    private data class EncodedInput(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenCount: Int,
    )

    private class BertWordPieceTokenizer(
        vocabFile: File,
        specialTokensFile: File,
        private val maxInputTokens: Int,
    ) {
        private val vocab: Map<String, Long> = vocabFile.useLines { lines ->
            lines.mapIndexed { index, token -> token to index.toLong() }.toMap()
        }
        private val specialTokens = JSONObject(specialTokensFile.readText())
        private val clsToken = specialTokens.getString("cls_token")
        private val sepToken = specialTokens.getString("sep_token")
        private val padToken = specialTokens.getString("pad_token")
        private val unkToken = specialTokens.getString("unk_token")
        private val neverSplit = setOf(clsToken, sepToken, padToken, unkToken, specialTokens.getString("mask_token"))
        private val padId = requireTokenId(padToken)
        private val unkId = requireTokenId(unkToken)

        fun encode(text: String): EncodedInput {
            val pieces = mutableListOf(clsToken)
            val maxWordPieces = (maxInputTokens - 2).coerceAtLeast(0)
            for (token in basicTokenize(text)) {
                if (pieces.size - 1 >= maxWordPieces) {
                    break
                }
                for (piece in wordPieceTokenize(token)) {
                    if (pieces.size - 1 >= maxWordPieces) {
                        break
                    }
                    pieces.add(piece)
                }
            }
            pieces.add(sepToken)

            val inputIds = LongArray(maxInputTokens) { padId }
            val attentionMask = LongArray(maxInputTokens)
            pieces.forEachIndexed { index, token ->
                inputIds[index] = vocab[token] ?: unkId
                attentionMask[index] = 1L
            }
            return EncodedInput(inputIds, attentionMask, pieces.size)
        }

        private fun basicTokenize(text: String): List<String> {
            val cleaned = addSpacesAroundChineseChars(cleanText(text))
            val outputTokens = mutableListOf<String>()
            for (token in whitespaceTokenize(cleaned)) {
                if (neverSplit.contains(token)) {
                    outputTokens.add(token)
                    continue
                }
                val normalized = stripAccents(token.lowercase(Locale.US))
                outputTokens.addAll(splitOnPunctuation(normalized))
            }
            return whitespaceTokenize(outputTokens.joinToString(" "))
        }

        private fun wordPieceTokenize(token: String): List<String> {
            if (token.length > 100) {
                return listOf(unkToken)
            }

            val subTokens = mutableListOf<String>()
            var start = 0
            while (start < token.length) {
                var end = token.length
                var current: String? = null
                while (start < end) {
                    val piece = if (start == 0) token.substring(start, end) else "##${token.substring(start, end)}"
                    if (vocab.containsKey(piece)) {
                        current = piece
                        break
                    }
                    end -= 1
                }
                if (current == null) {
                    return listOf(unkToken)
                }
                subTokens.add(current)
                start = end
            }
            return subTokens
        }

        private fun requireTokenId(token: String): Long {
            return vocab[token] ?: error("Required tokenizer token '$token' is missing from vocab.txt.")
        }

        private fun cleanText(text: String): String {
            val builder = StringBuilder()
            text.codePoints().forEach { codePoint ->
                when {
                    codePoint == 0 || codePoint == 0xfffd || isControl(codePoint) -> Unit
                    isWhitespace(codePoint) -> builder.append(' ')
                    else -> builder.appendCodePoint(codePoint)
                }
            }
            return builder.toString()
        }

        private fun addSpacesAroundChineseChars(text: String): String {
            val builder = StringBuilder()
            text.codePoints().forEach { codePoint ->
                if (isChineseChar(codePoint)) {
                    builder.append(' ')
                    builder.appendCodePoint(codePoint)
                    builder.append(' ')
                } else {
                    builder.appendCodePoint(codePoint)
                }
            }
            return builder.toString()
        }

        private fun stripAccents(text: String): String {
            val normalized = Normalizer.normalize(text, Normalizer.Form.NFD)
            val builder = StringBuilder()
            normalized.codePoints().forEach { codePoint ->
                if (Character.getType(codePoint) != Character.NON_SPACING_MARK.toInt()) {
                    builder.appendCodePoint(codePoint)
                }
            }
            return builder.toString()
        }

        private fun splitOnPunctuation(text: String): List<String> {
            val tokens = mutableListOf<String>()
            val builder = StringBuilder()
            text.codePoints().forEach { codePoint ->
                if (isPunctuation(codePoint)) {
                    if (builder.isNotEmpty()) {
                        tokens.add(builder.toString())
                        builder.setLength(0)
                    }
                    tokens.add(String(Character.toChars(codePoint)))
                } else {
                    builder.appendCodePoint(codePoint)
                }
            }
            if (builder.isNotEmpty()) {
                tokens.add(builder.toString())
            }
            return tokens
        }

        private fun whitespaceTokenize(text: String): List<String> {
            return text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        }

        private fun isWhitespace(codePoint: Int): Boolean {
            return Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint)
        }

        private fun isControl(codePoint: Int): Boolean {
            if (codePoint == '\t'.code || codePoint == '\n'.code || codePoint == '\r'.code) {
                return false
            }
            return when (Character.getType(codePoint)) {
                Character.CONTROL.toInt(), Character.FORMAT.toInt() -> true
                else -> false
            }
        }

        private fun isPunctuation(codePoint: Int): Boolean {
            if ((codePoint in 33..47) || (codePoint in 58..64) || (codePoint in 91..96) || (codePoint in 123..126)) {
                return true
            }
            return when (Character.getType(codePoint)) {
                Character.CONNECTOR_PUNCTUATION.toInt(),
                Character.DASH_PUNCTUATION.toInt(),
                Character.START_PUNCTUATION.toInt(),
                Character.END_PUNCTUATION.toInt(),
                Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
                Character.FINAL_QUOTE_PUNCTUATION.toInt(),
                Character.OTHER_PUNCTUATION.toInt() -> true
                else -> false
            }
        }

        private fun isChineseChar(codePoint: Int): Boolean {
            return codePoint in 0x4E00..0x9FFF ||
                codePoint in 0x3400..0x4DBF ||
                codePoint in 0x20000..0x2A6DF ||
                codePoint in 0x2A700..0x2B73F ||
                codePoint in 0x2B740..0x2B81F ||
                codePoint in 0x2B820..0x2CEAF ||
                codePoint in 0xF900..0xFAFF ||
                codePoint in 0x2F800..0x2FA1F
        }
    }
}
