"""Portable experiment identity, independent of paths and JSON formatting."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .data import FILES
from .io import ROOT, digest, read_json

CODE_FILES = (
    "data.py", "features.py", "temporal.py", "models.py", "metrics.py",
    "experiments.py", "identity.py", "io.py", "boost.py", "ensemble.py", "hgb_round.py",
)
PACKAGES = ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "optuna", "joblib", "threadpoolctl")


def logical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_hash(path):
    return logical_hash(read_json(path))


def text_hash(path):
    """For versioned text reports; original datasets retain their byte hashes."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def source_hash(root=ROOT):
    sources = {}
    for name in CODE_FILES:
        path = Path(root) / "src" / "bcp_datafest" / name
        # read_text uses universal newlines, so CRLF/LF have the same identity.
        sources[name] = path.read_text(encoding="utf-8")
    return logical_hash(sources)


def context(data_dir, protocol):
    return {
        "schema": "experiment-v2",
        "code_hash_schema": "normalized-source-v1",
        "code_sha256": source_hash(),
        "protocol_sha256": logical_hash(protocol),
        "protocol": protocol,
        "data_hashes": {name: digest(Path(data_dir) / name) for name in FILES},
        "python": ".".join(platform.python_version_tuple()[:2]),
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def experiment_identity(spec, months, source):
    return {"spec": spec, "months": list(months), "context": source}


def cache_matches(result, identity, prediction_path):
    return (result.get("identity") == identity
            and result.get("spec") == identity["spec"]
            and result.get("months") == identity["months"]
            and result.get("fingerprint") == logical_hash(identity)
            and Path(prediction_path).is_file()
            and result.get("predictions_sha256") == digest(prediction_path))


def compatible(result, source, months):
    identity = result.get("identity", {})
    return (identity.get("context") == source and identity.get("months") == list(months)
            and result.get("months") == list(months)
            and identity.get("spec") == result.get("spec")
            and result.get("fingerprint") == logical_hash(identity))
