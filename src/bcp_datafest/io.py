from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    # Windows may not ship an IANA timezone database. UTC−5 has no DST in Lima.
    from datetime import timezone, timedelta
    return datetime.now(timezone(timedelta(hours=-5))).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def markdown(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def code_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    h.update((ROOT / "configs/protocol.json").read_bytes())
    return h.hexdigest()


def provenance(data_dir):
    return {
        "timestamp_lima": now(), "code_sha256": code_hash(),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "optuna", "joblib"]},
        "data_hashes": {p.name: digest(p) for p in sorted(Path(data_dir).glob("*.csv"))},
        "protocol": read_json(ROOT / "configs/protocol.json"),
    }


def status(text):
    with (ROOT / "ESTADO.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()}\n\n{text}\n")
