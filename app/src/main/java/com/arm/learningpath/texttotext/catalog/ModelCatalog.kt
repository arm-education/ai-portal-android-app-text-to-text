package com.arm.learningpath.texttotext.catalog


import android.content.Context
import org.json.JSONArray

object ModelCatalog {
    fun load(context: Context): List<ModelConfig> {
        val catalogJson = context.assets.open("model_catalog.json").bufferedReader().use { it.readText() }
        val array = JSONArray(catalogJson)
        return List(array.length()) { index -> array.getJSONObject(index).toModelConfig() }
    }
}
