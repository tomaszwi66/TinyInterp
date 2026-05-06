import hashlib
import json

import numpy as np
import diskcache
import torch

import config

_cache = diskcache.Cache(str(config.CACHE_DIR))


def _serialize_value(v):
    if isinstance(v, torch.Tensor):
        return {"__tensor__": True, "data": v.cpu().numpy().tolist(), "dtype": str(v.dtype)}
    if isinstance(v, np.ndarray):
        return {"__ndarray__": True, "data": v.tolist(), "dtype": str(v.dtype)}
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def _deserialize_value(v):
    if isinstance(v, dict):
        if v.get("__tensor__"):
            return torch.tensor(v["data"])
        if v.get("__ndarray__"):
            return np.array(v["data"])
        return {k: _deserialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deserialize_value(item) for item in v]
    return v


def cache_result(key: str, result: dict, expire: int = 86400) -> None:
    serialized = _serialize_value(result)
    _cache.set(key, serialized, expire=expire)


def get_cached_result(key: str) -> dict | None:
    raw = _cache.get(key)
    if raw is None:
        return None
    return _deserialize_value(raw)


def make_cache_key(analysis_type: str, model_name: str, **params) -> str:
    parts = [analysis_type, model_name]
    for k in sorted(params.keys()):
        parts.append(f"{k}={params[k]}")
    raw = ":".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def clear_cache() -> None:
    _cache.clear()


def get_cache_size_mb() -> float:
    return _cache.volume() / (1024 * 1024)
