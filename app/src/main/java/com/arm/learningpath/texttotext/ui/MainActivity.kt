package com.arm.learningpath.texttotext.ui

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.text.TextUtils
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.PopupWindow
import android.widget.ScrollView
import android.widget.TextView
import com.arm.learningpath.texttotext.catalog.ModelCatalog
import com.arm.learningpath.texttotext.catalog.ModelConfig
import com.arm.learningpath.texttotext.inference.RuntimeRunner
import com.arm.learningpath.texttotext.inference.RuntimeRunnerFactory
import com.arm.learningpath.texttotext.storage.ModelStorage
import java.io.File

class MainActivity : Activity() {
    private val colorAccent = Color.rgb(0, 140, 145)
    private val colorTextPrimary = Color.rgb(24, 47, 55)
    private val colorTextSecondary = Color.rgb(85, 106, 114)
    private val colorSurface = Color.WHITE
    private val colorBackground = Color.rgb(246, 251, 250)
    private val colorBorder = Color.rgb(190, 217, 219)

    private lateinit var catalog: List<ModelConfig>
    private lateinit var modelDropdown: TextView
    private lateinit var promptInput: EditText
    private lateinit var statusView: TextView
    private lateinit var outputView: TextView
    private var runner: RuntimeRunner? = null
    private var loadedConfig: ModelConfig? = null
    private var selectedModelIndex: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        catalog = ModelCatalog.load(this)
        setContentView(createLayout())
        updateSelectedModelStatus()
    }

    override fun onDestroy() {
        runner?.close()
        super.onDestroy()
    }

    private fun createLayout(): View {
        val basePadding = dp(16)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(basePadding, basePadding, basePadding, basePadding)
            setBackgroundColor(colorBackground)
        }
        applySystemInsets(root, basePadding)

        root.addView(TextView(this).apply {
            text = "Text to Text Starter"
            textSize = 22f
            setTextColor(colorTextPrimary)
            setTypeface(typeface, Typeface.BOLD)
        })

        root.addView(TextView(this).apply {
            text = "Select a model catalog entry, load files from app-local storage, then run text generation or embeddings."
            textSize = 14f
            setTextColor(colorTextSecondary)
            setPadding(0, dp(6), 0, dp(16))
        })

        modelDropdown = TextView(this).apply {
            textSize = 18f
            setTextColor(colorTextPrimary)
            gravity = Gravity.CENTER_VERTICAL
            ellipsize = TextUtils.TruncateAt.END
            isSingleLine = true
            setPadding(dp(14), 0, dp(46), 0)
            text = modelLabel(catalog[selectedModelIndex])
        }
        val spinnerFrame = FrameLayout(this).apply {
            background = roundedBackground(fillColor = colorSurface, strokeColor = colorBorder)
            isClickable = true
            isFocusable = true
            setOnClickListener { showModelDropdown(this) }
            addView(modelDropdown, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ))
            addView(TextView(this@MainActivity).apply {
                text = "▼"
                textSize = 18f
                setTextColor(colorTextSecondary)
                gravity = Gravity.CENTER
                isClickable = false
            }, FrameLayout.LayoutParams(dp(42), FrameLayout.LayoutParams.MATCH_PARENT, Gravity.END))
        }
        root.addView(spinnerFrame, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)).apply {
            bottomMargin = dp(12)
        })

        root.addView(createActionButton("LOAD", filled = false).apply {
            setOnClickListener { loadSelectedModel() }
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)).apply {
            bottomMargin = dp(12)
        })

        promptInput = EditText(this).apply {
            hint = "Enter a prompt or text input"
            minLines = 4
            gravity = Gravity.TOP
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            setText("Write one sentence explaining why on-device AI is more private than cloud AI.")
            setTextColor(colorTextPrimary)
            setHintTextColor(colorTextSecondary)
            background = roundedBackground(fillColor = colorSurface, strokeColor = colorBorder)
            setPadding(dp(12), dp(10), dp(12), dp(10))
        }
        root.addView(promptInput, LinearLayout.LayoutParams.MATCH_PARENT, dp(132))

        root.addView(createActionButton("RUN", filled = true).apply {
            setOnClickListener { runSelectedModel() }
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)).apply {
            topMargin = dp(12)
            bottomMargin = dp(12)
        })

        root.addView(View(this).apply {
            setBackgroundColor(colorAccent)
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(2)).apply {
            bottomMargin = dp(14)
        })

        statusView = TextView(this).apply {
            textSize = 13f
            setTextColor(colorTextSecondary)
            setPadding(0, 0, 0, dp(10))
        }
        root.addView(statusView)

        outputView = TextView(this).apply {
            textSize = 14f
            setTextColor(colorTextPrimary)
            setTextIsSelectable(true)
        }

        root.addView(ScrollView(this).apply {
            addView(outputView)
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        return root
    }

    private fun applySystemInsets(root: View, basePadding: Int) {
        root.setOnApplyWindowInsetsListener { view, insets ->
            val topInset: Int
            val bottomInset: Int
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val systemInsets = insets.getInsets(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout(),
                )
                topInset = systemInsets.top
                bottomInset = systemInsets.bottom
            } else {
                @Suppress("DEPRECATION")
                topInset = insets.systemWindowInsetTop
                @Suppress("DEPRECATION")
                bottomInset = insets.systemWindowInsetBottom
            }

            view.setPadding(basePadding, basePadding + topInset, basePadding, basePadding + bottomInset)
            insets
        }
        root.requestApplyInsets()
        root.post { root.requestApplyInsets() }
    }

    private fun createActionButton(label: String, filled: Boolean): Button {
        return Button(this).apply {
            text = label
            textSize = 14f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(if (filled) Color.WHITE else colorAccent)
            background = roundedBackground(
                fillColor = if (filled) colorAccent else colorSurface,
                strokeColor = colorAccent,
            )
            minHeight = 0
            minWidth = 0
            includeFontPadding = false
        }
    }

    private fun roundedBackground(fillColor: Int, strokeColor: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dp(8).toFloat()
            setColor(fillColor)
            setStroke(dp(1), strokeColor)
        }
    }

    private fun modelLabel(config: ModelConfig): String {
        return if (config.displayName.contains("(${config.runtime})")) {
            config.displayName
        } else {
            "${config.displayName} - ${config.runtime}"
        }
    }

    private fun updateModelDropdownLabel() {
        if (::modelDropdown.isInitialized) {
            modelDropdown.text = modelLabel(catalog[selectedModelIndex])
        }
    }

    private fun showModelDropdown(anchor: View) {
        val labels = catalog.map { modelLabel(it) }
        var popupWindow: PopupWindow? = null
        val listView = ListView(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_list_item_1, labels)
            divider = null
            setOnItemClickListener { _, _, position, _ ->
                selectedModelIndex = position
                updateModelDropdownLabel()
                updateSelectedModelStatus()
                popupWindow?.dismiss()
            }
        }
        val maxHeight = dp(280)
        val rowHeight = dp(56)
        val popupHeight = (labels.size * rowHeight).coerceAtMost(maxHeight)
        popupWindow = PopupWindow(listView, anchor.width, popupHeight, true).apply {
            setBackgroundDrawable(roundedBackground(fillColor = colorSurface, strokeColor = colorBorder))
            elevation = dp(6).toFloat()
            isOutsideTouchable = true
            showAsDropDown(anchor, 0, dp(4))
        }
    }

    private fun loadSelectedModel() {
        val selected = selectedConfig()
        val modelDir = modelDir(selected)

        setBusy("Loading ${selected.displayName} from ${modelDir.absolutePath}")
        Thread {
            try {
                runner?.close()
                val nextRunner = RuntimeRunnerFactory.create(selected)
                val loadTimeMs = nextRunner.load(modelDir, selected)
                runner = nextRunner
                loadedConfig = selected
                showResult(
                    status = "Loaded ${selected.id} in ${loadTimeMs} ms",
                    output = "Ready. Runtime: ${selected.runtime}\nModel path: ${modelDir.absolutePath}",
                )
            } catch (error: Throwable) {
                showError("Load failed", error)
            }
        }.start()
    }

    private fun runSelectedModel() {
        val selected = selectedConfig()
        val activeRunner = runner
        if (activeRunner == null || loadedConfig?.id != selected.id) {
            outputView.text = "Load the selected model before running it."
            return
        }

        val input = promptInput.text.toString()
        setBusy("Running ${selected.displayName}")
        Thread {
            try {
                val result = if (selected.requiresEmbedding) {
                    activeRunner.runEmbedding(input)
                } else {
                    activeRunner.runTextGeneration(input)
                }
                showResult(
                    status = "Load: ${result.loadTimeMs} ms | Run: ${result.runTimeMs} ms",
                    output = result.text,
                )
            } catch (error: Throwable) {
                showError("Run failed", error)
            }
        }.start()
    }

    private fun selectedConfig(): ModelConfig {
        return catalog[selectedModelIndex.coerceIn(catalog.indices)]
    }

    private fun modelDir(config: ModelConfig): File {
        return ModelStorage.modelDir(this, config)
    }

    private fun updateSelectedModelStatus() {
        if (!::statusView.isInitialized) {
            return
        }

        val selected = selectedConfig()
        val dir = modelDir(selected)
        if (selected.runtime == "mock") {
            statusView.text = """
                Runtime: ${selected.runtime}
                Workload: ${selected.workload}
                Mock runner: no local model files are required.
                App-local model directory is ignored for this catalog entry.
            """.trimIndent()
            return
        }

        val modelFile = File(dir, selected.modelFile)
        val state = if (modelFile.exists()) "found" else "missing"
        statusView.text = """
            Runtime: ${selected.runtime}
            Workload: ${selected.workload}
            App-local model directory: ${dir.absolutePath}
            Required model file: ${modelFile.name} ($state)
        """.trimIndent()
    }

    private fun setBusy(message: String) {
        runOnUiThread {
            statusView.text = message
            outputView.text = "Working..."
        }
    }

    private fun showResult(status: String, output: String) {
        runOnUiThread {
            statusView.text = status
            outputView.text = output
        }
    }

    private fun showError(title: String, error: Throwable) {
        runOnUiThread {
            statusView.text = title
            outputView.text = error.message ?: error.toString()
        }
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }
}
