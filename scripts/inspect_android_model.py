#!/usr/bin/env python3
"""Create local model context for the Android text-to-text starter app."""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_EXTENSIONS = {
    "executorch": [".pte"],
    "litert": [".tflite"],
    "tflite": [".tflite"],
    "litert-lm": [".litertlm"],
    "onnx": [".onnx"],
    "onnxruntime": [".onnx"],
    "onnxruntime-genai": [".onnx"],
}

TOKENIZER_NAMES = {
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
}

CONFIG_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".jinja",
    ".py",
}

METADATA_NAMES = {
    "config.json",
    "config.yaml",
    "config.yml",
    "metadata.json",
    "metadata.yaml",
    "metadata.yml",
    "embeddings.json",
}

MAX_METADATA_BYTES = 512 * 1024
IGNORED_SUFFIXES = {
    ".lock",
    ".metadata",
    ".incomplete",
}


def should_ignore_path(path: Path) -> bool:
    parts = path.parts
    if any(part.startswith(".") for part in parts):
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    if parts and parts[0] == "benchmarks":
        return True
    if path.name == "__pycache__":
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local Android model folder and generate context for a runtime adapter."
    )
    parser.add_argument("--model-id", required=True, help="Stable ID used in model_catalog.json.")
    parser.add_argument("--model-source", required=True, help="Model repository URL, model card URL, or artifact source.")
    parser.add_argument("--runtime", required=True, help="Runtime value, such as executorch, litert, litert-lm, or onnxruntime.")
    parser.add_argument("--workload", required=True, choices=["text-generation", "text-embedding"])
    parser.add_argument("--local-model-dir", required=True, type=Path, help="Directory containing downloaded model files.")
    parser.add_argument(
        "--target-device",
        default="Arm-based Android device",
        help="Optional device description recorded as metadata only.",
    )
    parser.add_argument("--output-dir", default=Path("."), type=Path, help="Starter app project root.")
    return parser.parse_args()


def relative_files(model_dir: Path) -> list[dict[str, object]]:
    files = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        if should_ignore_path(Path(relative)):
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "extension": path.suffix.lower(),
                "mime_type": mimetypes.guess_type(path.name)[0] or "",
            }
        )
    return files


def select_candidates(files: list[dict[str, object]], runtime: str) -> list[str]:
    extensions = RUNTIME_EXTENSIONS.get(runtime, [])
    if runtime in {"onnxruntime-genai", "onnxruntime"}:
        config_paths = [item["path"] for item in files if item["path"].endswith("genai_config.json")]
        if config_paths:
            return config_paths
    return [str(item["path"]) for item in files if str(item["extension"]) in extensions]


def find_support_files(files: list[dict[str, object]]) -> dict[str, list[str]]:
    tokenizer = []
    templates = []
    configs = []
    external_data = []

    for item in files:
        path = str(item["path"])
        name = Path(path).name
        extension = str(item["extension"])
        if name in TOKENIZER_NAMES:
            tokenizer.append(path)
        if extension == ".jinja" or "template" in name.lower():
            templates.append(path)
        if extension in CONFIG_SUFFIXES and path not in tokenizer and path not in templates:
            configs.append(path)
        if extension in {".data", ".bin"} or path.endswith(".onnx.data"):
            external_data.append(path)

    return {
        "tokenizer_files": sorted(set(tokenizer)),
        "prompt_template_files": sorted(set(templates)),
        "configuration_files": sorted(set(configs)),
        "external_data_files": sorted(set(external_data)),
    }


def copy_metadata(model_dir: Path, context_dir: Path, files: list[dict[str, object]]) -> list[str]:
    metadata_dir = context_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for item in files:
        path = str(item["path"])
        size = int(item["size_bytes"])
        extension = str(item["extension"])
        if size > MAX_METADATA_BYTES or extension not in CONFIG_SUFFIXES:
            continue

        source = model_dir / path
        destination = metadata_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(path)

    return copied


def parse_yaml_metadata(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:
        return parse_simple_yaml(text)
    return yaml.safe_load(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for simple metadata files when PyYAML is unavailable."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        stripped = line_without_comment.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if pending_key and indent > pending_key[0]:
            _, pending_parent, pending_name = pending_key
            if stripped.startswith("- "):
                new_child: Any = []
            else:
                new_child = {}
            pending_parent[pending_name] = new_child
            stack.append((indent - 1, new_child))
            parent = new_child
            pending_key = None

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                continue
            item_text = stripped[2:].strip()
            if ":" in item_text:
                key, value = item_text.split(":", 1)
                item: dict[str, Any] = {key.strip(): parse_scalar(value.strip())}
                parent.append(item)
                stack.append((indent, item))
                if value.strip() == "":
                    pending_key = (indent, item, key.strip())
            else:
                parent.append(parse_scalar(item_text))
            continue

        if ":" not in stripped or not isinstance(parent, dict):
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
        else:
            parent[key] = {}
            pending_key = (indent, parent, key)

    return root


def parse_scalar(value: str) -> Any:
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [parse_scalar(part.strip()) for part in value[1:-1].split(",") if part.strip()]
    quoted = value.strip("\"'")
    if quoted != value:
        return quoted
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_metadata_documents(model_dir: Path, files: list[dict[str, object]]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for item in files:
        relative = str(item["path"])
        name = Path(relative).name
        if name not in METADATA_NAMES and relative not in {"config.yaml", "metadata.yaml", "embeddings.json"}:
            continue
        if int(item["size_bytes"]) > MAX_METADATA_BYTES:
            continue

        source = model_dir / relative
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        try:
            if source.suffix.lower() == ".json":
                parsed = json.loads(text)
            elif source.suffix.lower() in {".yaml", ".yml"}:
                parsed = parse_yaml_metadata(text)
            else:
                continue
        except Exception as error:
            documents[relative] = {"_parse_error": str(error)}
            continue

        if parsed is not None:
            documents[relative] = parsed
    return documents


def walk_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(walk_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(walk_values(child))
    return values


def iter_key_values(value: Any) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            pairs.append((str(key), child))
            pairs.extend(iter_key_values(child))
    elif isinstance(value, list):
        for child in value:
            pairs.extend(iter_key_values(child))
    return pairs


def find_first_string_for_keys(documents: dict[str, Any], keys: set[str]) -> str:
    for document in documents.values():
        for key, value in iter_key_values(document):
            if key.lower() in keys and isinstance(value, str) and value:
                return value
    return ""


def find_model_file_from_metadata(candidates: list[str], documents: dict[str, Any]) -> str:
    filename = find_first_string_for_keys(documents, {"filename", "file_name", "model_file", "model_filename"})
    if not filename:
        return ""

    normalized = filename.replace("\\", "/")
    for candidate in candidates:
        if candidate == normalized or Path(candidate).name == Path(normalized).name:
            return candidate
    return ""


def find_runtime_config(support: dict[str, list[str]]) -> str:
    configs = support["configuration_files"]
    for preferred in ("config.yaml", "config.yml", "genai_config.json"):
        for config in configs:
            if config.endswith(preferred):
                return config
    return configs[0] if len(configs) == 1 else ""


def shape_to_ints(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, int) and item > 0:
            result.append(item)
        elif isinstance(item, str) and item.isdigit():
            result.append(int(item))
    return result


def find_max_input_tokens(documents: dict[str, Any]) -> int:
    preferred_keys = {"max_length", "max_seq_len", "max_sequence_length", "max_input_tokens", "sequence_length"}
    for document in documents.values():
        for key, value in iter_key_values(document):
            if key.lower() in preferred_keys and isinstance(value, int) and value > 0:
                return value

    for document in documents.values():
        for value in walk_values(document):
            if not isinstance(value, dict):
                continue
            name = str(value.get("name", "")).lower()
            shape = shape_to_ints(value.get("shape"))
            if name in {"input_ids", "attention_mask"} and len(shape) >= 2:
                return shape[-1]
    return 0


def find_embedding_dimensions(documents: dict[str, Any]) -> int:
    preferred_keys = {"embedding_dimensions", "embedding_dimension", "embedding_dim", "dimensions", "dimension", "dim"}
    for document in documents.values():
        for key, value in iter_key_values(document):
            key_lower = key.lower()
            if key_lower in preferred_keys and isinstance(value, int) and value > 0:
                return value

    for name, document in documents.items():
        if Path(name).name == "embeddings.json":
            ints = [value for value in walk_values(document) if isinstance(value, int) and value > 0]
            if ints:
                return max(ints)

    for document in documents.values():
        for value in walk_values(document):
            if not isinstance(value, dict):
                continue
            name = str(value.get("name", "")).lower()
            shape = shape_to_ints(value.get("shape"))
            if ("embedding" in name or name in {"output", "sentence_embedding"}) and shape:
                return shape[-1]
    return 0


def collect_tensor_names(documents: dict[str, Any], section_name: str) -> list[str]:
    names: list[str] = []
    section_name = section_name.lower()
    for document in documents.values():
        for key, value in iter_key_values(document):
            if key.lower() != section_name:
                continue
            for child in walk_values(value):
                if isinstance(child, dict) and isinstance(child.get("name"), str):
                    names.append(child["name"])
                elif isinstance(child, dict) and isinstance(child.get("fields"), list):
                    for field in child["fields"]:
                        if isinstance(field, dict) and isinstance(field.get("name"), str):
                            names.append(field["name"])
            if isinstance(value, dict) and isinstance(value.get("fields"), list):
                for field in value["fields"]:
                    if isinstance(field, dict) and isinstance(field.get("name"), str):
                        names.append(field["name"])
    return sorted(dict.fromkeys(names))


def find_tokenizer_files(files: list[dict[str, object]], support: dict[str, list[str]]) -> list[str]:
    tokenizer_files = set(support["tokenizer_files"])
    for item in files:
        path = str(item["path"])
        name = Path(path).name.lower()
        parent = Path(path).parent.as_posix().lower()
        if name in {tokenizer.lower() for tokenizer in TOKENIZER_NAMES}:
            tokenizer_files.add(path)
        elif "tokenizer" in parent and str(item["extension"]) in CONFIG_SUFFIXES:
            tokenizer_files.add(path)
    return sorted(tokenizer_files)


def build_catalog_suggestion(
    args: argparse.Namespace,
    candidates: list[str],
    support: dict[str, list[str]],
    metadata_documents: dict[str, Any],
    files: list[dict[str, object]],
) -> dict[str, object]:
    metadata_primary = find_model_file_from_metadata(candidates, metadata_documents)
    primary = metadata_primary or (candidates[0] if len(candidates) == 1 else "")
    if args.runtime in {"onnxruntime-genai", "onnxruntime"} and primary.endswith("genai_config.json"):
        model_file = "model.onnx"
        runtime_config = primary
    else:
        model_file = primary
        runtime_config = find_runtime_config(support)

    return {
        "id": args.model_id,
        "displayName": "Text generation model" if args.workload == "text-generation" else "Text embedding model",
        "workload": args.workload,
        "runtime": args.runtime,
        "artifactType": "file" if model_file else "directory",
        "artifactPath": args.model_id,
        "modelFile": model_file,
        "externalDataFiles": support["external_data_files"],
        "runtimeConfig": runtime_config,
        "tokenizerFiles": find_tokenizer_files(files, support),
        "chatTemplate": support["prompt_template_files"][0] if support["prompt_template_files"] else "",
        "maxInputTokens": find_max_input_tokens(metadata_documents),
        "maxNewTokens": 64 if args.workload == "text-generation" else 0,
        "embeddingDimensions": find_embedding_dimensions(metadata_documents),
        "normalizeEmbeddings": args.workload == "text-embedding",
        "inputTensorNames": collect_tensor_names(metadata_documents, "input"),
        "outputTensorNames": collect_tensor_names(metadata_documents, "output"),
    }


def main() -> int:
    args = parse_args()
    model_dir = args.local_model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    files = relative_files(model_dir)
    support = find_support_files(files)
    candidates = select_candidates(files, args.runtime)
    metadata_documents = load_metadata_documents(model_dir, files)
    context_dir = output_dir / "model-context"
    if context_dir.exists():
        shutil.rmtree(context_dir)
    copied_metadata = copy_metadata(model_dir, context_dir, files)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_source": args.model_source,
        "runtime": args.runtime,
        "workload": args.workload,
        "target_device": args.target_device,
        "local_model_dir": str(model_dir),
        "files": files,
        "primary_artifact_candidates": candidates,
        "support_files": support,
        "parsed_metadata_files": sorted(metadata_documents.keys()),
        "copied_metadata_files": copied_metadata,
        "notes": [
            "Review primary_artifact_candidates. If more than one candidate is listed, the catalog suggestion uses metadata.yaml filename when it matches one artifact; otherwise choose the intended Android artifact manually.",
            "Fill missing token limits, tensor names, embedding dimensions, device requirements, and runtime dependency from the model card or runtime documentation.",
            "Do not put credentials in the Android app or model catalog.",
        ],
    }

    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "model-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    config = {
        "model_id": args.model_id,
        "model_source": args.model_source,
        "runtime": args.runtime,
        "workload": args.workload,
        "target_device": args.target_device,
        "local_model_dir": str(model_dir),
        "catalog_suggestion": build_catalog_suggestion(args, candidates, support, metadata_documents, files),
    }
    (output_dir / "android_model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_dir / 'android_model_config.json'}")
    print(f"Wrote {context_dir / 'model-summary.json'}")
    print(f"Copied {len(copied_metadata)} metadata files to {context_dir / 'metadata'}")
    if len(candidates) == 0:
        print("No primary artifact candidate found for the selected runtime.")
    elif len(candidates) > 1:
        print("Multiple primary artifact candidates found:")
        for index, candidate in enumerate(candidates, start=1):
            print(f"  {index}. {candidate}")
    else:
        print(f"Primary artifact candidate: {candidates[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
