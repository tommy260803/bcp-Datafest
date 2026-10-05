"""Regenerate December from the unchanged A recipe, without altering its history."""
from __future__ import annotations

import platform
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from .boost import candidate_a_reference
from .data import load
from .features import make_features
from .identity import context, json_hash, logical_hash
from .io import ROOT, digest, now, read_json, write_json
from .models import build, fit_model
from .submission import validate_submission


def verify_models(metadata, recipe, frozen):
    if metadata["recipe"] != recipe or metadata["fingerprint"] != logical_hash(recipe):
        raise ValueError("Saved final models do not match the current A recipe/code/data")
    if len(metadata["components"]) != len(frozen["components"]):
        raise ValueError("Incomplete A model bundle")
    for entry, component in zip(metadata["components"], frozen["components"]):
        if (entry["spec"] != component["spec"] or entry["weight"] != component["weight"]
                or entry["columns"] != frozen["feature_columns"][component["spec"]["name"]]):
            raise ValueError("Saved A component configuration or columns changed")
        path = ROOT / entry["path"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError("Saved A model is missing or damaged")


def export_submission(output, test, probabilities, sample, reports):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        pd.DataFrame({"id_cliente": test.id_cliente.to_numpy(), "prediccion": probabilities}).to_csv(
            temporary, index=False, encoding="utf-8", float_format="%.15g", lineterminator="\n")
        if output.exists():
            if digest(output) != digest(temporary):
                raise FileExistsError(f"Existing submission differs; choose another --output: {output}")
        else:
            temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return validate_submission(output, test, sample, report_dir=reports)


def run(data_dir, output):
    frozen = candidate_a_reference(data_dir)
    original_protocol = frozen["provenance"]["protocol"]
    source = context(data_dir, original_protocol)
    recipe = {"schema": "delivery-a-v1", "candidate": "A",
              "config_logical_sha256": json_hash(ROOT / "configs/final.json"), "context": source}
    fingerprint = logical_hash(recipe)
    train, test, sample, _ = load(data_dir)
    if (sorted(train.mes.unique().tolist()) != list(range(202601, 202612))
            or not test.mes.eq(202612).all() or len(train) != 110100):
        raise ValueError("A delivery requires all January-November labels and the December test")
    weights = [c["weight"] for c in frozen["components"]]
    if not all(np.isfinite(w) and w >= 0 for w in weights) or abs(sum(weights) - 1) > 1e-10:
        raise ValueError("Invalid frozen ensemble weights")
    threads = original_protocol["threads"]
    directory = ROOT / "artifacts/boost/delivery_a/models" / fingerprint
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        verify_models(metadata, recipe, frozen)
        print("Reusing saved A final models", flush=True)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        for i, component in enumerate(frozen["components"]):
            spec = component["spec"]
            x = make_features(train, spec["blocks"])
            if list(x.columns) != frozen["feature_columns"][spec["name"]]:
                raise ValueError("A feature schema changed")
            if spec["family"] not in ("catboost", "hgb"):
                raise ValueError("This delivery reproduces only the frozen CatBoost/HGB A")
            model = build(spec, x, threads)
            start = time.perf_counter()
            with threadpool_limits(limits=threads):
                fit_model(model, spec, x, train.objetivo)
            path = directory / f"component_{i}{'.cbm' if spec['family'] == 'catboost' else '.joblib'}"
            if spec["family"] == "catboost":
                model.save_model(str(path))
            else:
                joblib.dump(model, path)
            entries.append({"spec": spec, "weight": component["weight"], "columns": list(x.columns),
                            "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                            "seconds": time.perf_counter() - start})
            print(f"A final fit {spec['name']}: {entries[-1]['seconds']:.1f}s; {len(train)} rows", flush=True)
        metadata = {"recipe": recipe, "fingerprint": fingerprint, "components": entries,
                    "train_rows": len(train), "train_months": sorted(train.mes.unique().tolist()),
                    "runtime": {"timestamp_lima": now(), "python": platform.python_version(), "platform": platform.platform()}}
        write_json(metadata_path, metadata)
    verify_models(metadata, recipe, frozen)
    if metadata["train_rows"] != len(train) or metadata["train_months"] != sorted(train.mes.unique().tolist()):
        raise ValueError("Final A model training period changed")
    combined = pd.concat([train.drop(columns="objetivo"), test], ignore_index=True)
    positions = np.arange(len(train), len(combined))
    p = np.zeros(len(test))
    for entry in metadata["components"]:
        x = make_features(combined, entry["spec"]["blocks"]).iloc[positions]
        if list(x.columns) != entry["columns"]:
            raise ValueError("A inference feature schema changed")
        path = ROOT / entry["path"]
        if entry["spec"]["family"] == "catboost":
            model = CatBoostClassifier()
            model.load_model(str(path))
        else:
            model = joblib.load(path)
        with threadpool_limits(limits=threads):
            p += entry["weight"] * model.predict_proba(x)[:, 1]
    if (context(data_dir, original_protocol) != source or not np.isfinite(p).all()
            or not ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Delivery identity changed or probabilities are invalid")
    if not combined.iloc[positions].id_cliente.reset_index(drop=True).equals(test.id_cliente):
        raise ValueError("December input order changed")
    reports = ROOT / "reports/boost/delivery_a"
    validation = export_submission(output, test, p, sample, reports)
    result = {"candidate": "A", "recipe": recipe, "fingerprint": fingerprint,
              "model_metadata_path": metadata_path.relative_to(ROOT).as_posix(),
              "model_metadata_sha256": digest(metadata_path), "validation": validation,
              "training_months": metadata["train_months"], "prediction_month": 202612,
              "development_mean_gini": frozen["mean_gini"],
              "november_gini_historical": read_json(ROOT / "reports/final_validation.json")["metrics"]["gini"],
              "december_gini": None, "november_reevaluated": False}
    write_json(reports / "prediction_provenance.json", result)
    print(validation, flush=True)
    return result
