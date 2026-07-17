#!/usr/bin/env python3
"""Readable folder-based cache helpers for d-vector extraction artifacts."""

import json
import shutil
import time
import hashlib
from pathlib import Path

import numpy as np


CACHE_LAYOUT_VERSION = 1

_CACHE_NAMES_DROP_RANDOM_SEED = {
    "clean_target_plus_noise_pairs",
    "overlap_non_target_chunked_pairs",
    "clean_target_identity_chunked_pairs",
    "clean_non_target_identity_chunked_pairs",
    "speech_noise_dvectors",
}


def _jsonify(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, set):
        return [_jsonify(item) for item in sorted(value, key=lambda item: str(item))]
    return value


def normalize_for_hash(value):
    """Normalize nested config payloads into deterministic JSON-safe structures."""
    if isinstance(value, Path):
        return str(value.resolve())

    if isinstance(value, dict):
        normalized = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            normalized[str(key)] = normalize_for_hash(value[key])
        return normalized

    if isinstance(value, (list, tuple)):
        return [normalize_for_hash(item) for item in value]

    if isinstance(value, set):
        return [normalize_for_hash(item) for item in sorted(value, key=lambda item: str(item))]

    if isinstance(value, np.generic):
        return value.item()

    return value


def canonicalize_cache_config(cache_name, cache_config):
    """Strip non-data settings so cache keys follow the underlying data."""
    canonical = dict(cache_config or {})

    canonical.pop("device", None)
    canonical.pop("pair_target_count", None)

    if cache_name in _CACHE_NAMES_DROP_RANDOM_SEED:
        canonical.pop("random_seed", None)

    return canonical


def _cache_identity(cache_name, cache_config, cache_version):
    canonical_config = canonicalize_cache_config(cache_name, cache_config)
    normalized_config = normalize_for_hash(canonical_config)
    payload = {
        "cache_name": cache_name,
        "cache_version": cache_version,
        "config": normalized_config,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return normalized_config, digest


def cache_entry_dir(cache_dir, cache_name, cache_config, cache_version=1):
    _, digest = _cache_identity(cache_name, cache_config, cache_version)
    return Path(cache_dir) / cache_name / digest


def checkpoint_dir_for_config(cache_dir, cache_name, cache_config, cache_version=1):
    return cache_entry_dir(cache_dir, cache_name, cache_config, cache_version) / "_checkpoint"


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_pair_dataset(data_dir, data):
    np.save(data_dir / "noisy_dvectors.npy", np.asarray(data["noisy_dvectors"]))
    np.save(data_dir / "clean_dvectors.npy", np.asarray(data["clean_dvectors"]))
    _save_json(data_dir / "keys.json", list(data.get("keys", [])))
    _save_json(data_dir / "metadata.json", _jsonify(list(data.get("metadata", []))))

    layout = {
        "kind": "paired_dvectors",
        "count": int(len(data.get("keys", []))),
    }

    preview_items = list(data.get("preview_items", []))
    if preview_items:
        preview_meta = []
        preview_audio = []
        for item in preview_items:
            item_meta = {}
            for key, value in item.items():
                if key == "audio":
                    preview_audio.append(np.asarray(value, dtype=np.float32))
                else:
                    item_meta[key] = _jsonify(value)
            preview_meta.append(item_meta)

        _save_json(data_dir / "preview_items.json", preview_meta)
        if preview_audio:
            np.save(data_dir / "preview_audio.npy", np.stack(preview_audio).astype(np.float32))
        layout["preview_count"] = len(preview_items)

    return layout


def _load_pair_dataset(data_dir):
    payload = {
        "keys": _load_json(data_dir / "keys.json"),
        "noisy_dvectors": np.load(data_dir / "noisy_dvectors.npy"),
        "clean_dvectors": np.load(data_dir / "clean_dvectors.npy"),
        "metadata": _load_json(data_dir / "metadata.json"),
    }

    preview_meta_path = data_dir / "preview_items.json"
    preview_audio_path = data_dir / "preview_audio.npy"
    if preview_meta_path.exists():
        preview_items = _load_json(preview_meta_path)
        if preview_audio_path.exists():
            preview_audio = np.load(preview_audio_path)
            for index, audio in enumerate(preview_audio):
                preview_items[index]["audio"] = audio
        payload["preview_items"] = preview_items
    else:
        payload["preview_items"] = []

    return payload


def _save_pair_list(data_dir, pairs):
    pairs = list(pairs)
    if pairs:
        inputs = np.stack([np.asarray(pair[0], dtype=np.float32) for pair in pairs]).astype(np.float32)
        targets = np.stack([np.asarray(pair[1], dtype=np.float32) for pair in pairs]).astype(np.float32)
    else:
        inputs = np.empty((0, 0), dtype=np.float32)
        targets = np.empty((0, 0), dtype=np.float32)

    np.save(data_dir / "inputs.npy", inputs)
    np.save(data_dir / "targets.npy", targets)
    return {"kind": "pair_list", "count": int(len(pairs))}


def _load_pair_list(data_dir):
    inputs = np.load(data_dir / "inputs.npy")
    targets = np.load(data_dir / "targets.npy")
    return [(inputs[index], targets[index]) for index in range(len(inputs))]


def _collect_dvector_entries(node, path=()):
    if isinstance(node, dict) and "dvector" in node:
        entry = dict(node)
        entry["dvector"] = np.asarray(entry["dvector"], dtype=np.float32)
        return [(list(path), entry)]

    if not isinstance(node, dict):
        return None

    entries = []
    for key in sorted(node.keys(), key=lambda item: str(item)):
        child_entries = _collect_dvector_entries(node[key], path + (str(key),))
        if child_entries is None:
            return None
        entries.extend(child_entries)
    return entries


def _save_dvector_map(data_dir, data):
    entries = _collect_dvector_entries(data)
    if entries is None:
        raise TypeError("Unsupported cache payload shape")

    if entries:
        dvectors = np.stack([entry["dvector"] for _, entry in entries]).astype(np.float32)
    else:
        dvectors = np.empty((0, 0), dtype=np.float32)

    serialized_entries = []
    for path, entry in entries:
        serialized_entries.append(
            {
                "path": path,
                "metadata": _jsonify({key: value for key, value in entry.items() if key != "dvector"}),
            }
        )

    np.save(data_dir / "dvectors.npy", dvectors)
    _save_json(data_dir / "entries.json", serialized_entries)
    return {"kind": "dvector_map", "count": int(len(entries))}


def _load_dvector_map(data_dir):
    dvectors = np.load(data_dir / "dvectors.npy")
    entries = _load_json(data_dir / "entries.json")
    root = {}

    for index, entry in enumerate(entries):
        path = list(entry["path"])
        if not path:
            continue

        current = root
        for token in path[:-1]:
            current = current.setdefault(token, {})

        leaf = dict(entry["metadata"])
        leaf["dvector"] = dvectors[index]
        current[path[-1]] = leaf

    return root


def _detect_layout(data):
    if isinstance(data, dict) and "noisy_dvectors" in data and "clean_dvectors" in data:
        return "paired_dvectors"
    if isinstance(data, (list, tuple)):
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in data):
            return "pair_list"
    if isinstance(data, dict):
        entries = _collect_dvector_entries(data)
        if entries is not None:
            return "dvector_map"
    raise TypeError(f"Unsupported cache payload type for readable storage: {type(data)!r}")


def _save_cache_data(data_dir, data):
    layout_kind = _detect_layout(data)
    if layout_kind == "paired_dvectors":
        return _save_pair_dataset(data_dir, data)
    if layout_kind == "pair_list":
        return _save_pair_list(data_dir, data)
    if layout_kind == "dvector_map":
        return _save_dvector_map(data_dir, data)
    raise TypeError(f"Unsupported cache layout kind: {layout_kind}")


def _load_cache_data(data_dir, layout):
    kind = layout["kind"]
    if kind == "paired_dvectors":
        return _load_pair_dataset(data_dir)
    if kind == "pair_list":
        return _load_pair_list(data_dir)
    if kind == "dvector_map":
        return _load_dvector_map(data_dir)
    raise TypeError(f"Unsupported cache layout kind: {kind}")


def save_cache_entry(cache_dir, cache_name, cache_config, data, cache_version=1):
    normalized_config, _ = _cache_identity(cache_name, cache_config, cache_version)
    entry_dir = cache_entry_dir(cache_dir, cache_name, cache_config, cache_version)
    data_dir = entry_dir / "data"
    tmp_dir = entry_dir.parent / f"{entry_dir.name}.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    data_dir_tmp = tmp_dir / "data"
    data_dir_tmp.mkdir(parents=True, exist_ok=True)

    layout = _save_cache_data(data_dir_tmp, data)
    manifest = {
        "layout_version": CACHE_LAYOUT_VERSION,
        "cache_name": cache_name,
        "cache_version": cache_version,
        "config": normalized_config,
        "layout": layout,
    }
    _save_json(tmp_dir / "manifest.json", manifest)

    if entry_dir.exists():
        shutil.rmtree(entry_dir)
    tmp_dir.replace(entry_dir)
    return entry_dir


def load_cache_entry(cache_dir, cache_name, cache_config, cache_version=1):
    normalized_config, _ = _cache_identity(cache_name, cache_config, cache_version)
    entry_dir = cache_entry_dir(cache_dir, cache_name, cache_config, cache_version)
    manifest_path = entry_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = _load_json(manifest_path)
    if manifest.get("cache_version") != cache_version:
        return None
    if manifest.get("config") != normalized_config:
        return None

    return _load_cache_data(entry_dir / "data", manifest["layout"])


def extract_with_cache(cache_dir, cache_name, cache_config, extractor_fn, cache_version=1):
    """Load a readable cache folder by config, or extract then save."""
    t_start = time.perf_counter()
    cache_dir = Path(cache_dir)

    cached_data = load_cache_entry(cache_dir, cache_name, cache_config, cache_version=cache_version)
    if cached_data is not None:
        n_items = 0
        if isinstance(cached_data, dict) and "keys" in cached_data:
            n_items = len(cached_data.get("keys", []))
        elif isinstance(cached_data, (list, tuple)):
            n_items = len(cached_data)
        elif isinstance(cached_data, dict):
            n_items = len(cached_data)
        print(f"[cache] Hit for {cache_name}: {cache_entry_dir(cache_dir, cache_name, cache_config, cache_version)} ({n_items} items)")
        print(f"[perf] Cache load time: {time.perf_counter() - t_start:.2f}s")
        return cached_data

    print(f"[cache] Miss for {cache_name}; extracting")
    data = extractor_fn()
    entry_dir = save_cache_entry(cache_dir, cache_name, cache_config, data, cache_version=cache_version)
    print(f"[cache] Saved: {entry_dir}")
    print(f"[perf] Total cache wrapper time: {time.perf_counter() - t_start:.2f}s")
    return data
