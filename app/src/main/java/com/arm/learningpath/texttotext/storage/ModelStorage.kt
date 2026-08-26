package com.arm.learningpath.texttotext.storage

import android.content.Context
import com.arm.learningpath.texttotext.catalog.ModelConfig
import java.io.File

object ModelStorage {
    fun modelDir(context: Context, config: ModelConfig): File {
        return File(context.filesDir, "models/${config.id}")
    }
}
