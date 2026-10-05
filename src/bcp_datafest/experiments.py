from __future__ import annotations

import time
import shutil
import uuid
import warnings
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load
from .features import make_features
from .io import ROOT, read_json, write_json, digest, markdown, status, now
from .metrics import metrics
from .models import build, fit_model
from .identity import context, experiment_identity, logical_hash, cache_matches, compatible


def locations(workspace=None, round_name=None):
    if workspace is not None and (not workspace or any(c in workspace for c in "/\\.")):
        raise ValueError("Workspace must be a simple name")
    suffix = Path(workspace) if workspace else Path()
    if round_name is not None:
        if not workspace or not round_name or any(c in round_name for c in "/\\."):
            raise ValueError("A round requires a workspace and a simple name")
        runs = ROOT / "artifacts" / suffix / "rounds" / round_name / "runs"
        reports = ROOT / "reports" / suffix / "rounds" / round_name
        return ROOT / "configs" / suffix / "protocol.json", runs, reports
    return (ROOT / "configs" / suffix / "protocol.json",
            ROOT / "artifacts" / suffix / "runs", ROOT / "reports" / suffix)


def evaluate(spec, data_dir, months=None, *, workspace=None, round_config=None):
    protocol_path, runs, _ = locations(workspace, None if round_config is None else round_config["name"])
    protocol = read_json(protocol_path)
    if round_config is not None:
        protocol["round_config"] = round_config
    months = list(protocol["development_months"] if months is None else months)
    if not months or len(set(months)) != len(months):
        raise ValueError("Evaluation requires distinct, nonempty months")
    if workspace == "boost" and (months != protocol["development_months"]
                                 or any(m not in (202608, 202609, 202610) for m in months)):
        raise ValueError("Boost experiments are restricted to development months")
    if any(c in spec["name"] for c in "/\\") or spec["name"] in ("", ".", ".."):
        raise ValueError("Run name must be a simple identifier")
    source = context(data_dir, protocol)
    identity = experiment_identity(spec, months, source)
    fingerprint = logical_hash(identity)
    run = runs / spec["name"]
    if workspace:
        run /= fingerprint
    if (run / "result.json").exists():
        previous = read_json(run / "result.json")
        if cache_matches(previous, identity, run / "validation.csv"):
            return previous
        archive = runs.parent / "archive" / f"{spec['name']}_{uuid.uuid4().hex}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), str(archive))
        warnings.warn(f"Incompatible/incomplete cache archived at {archive}; recalculating", stacklevel=2)
    train, test, _, _ = load(data_dir)
    if any(not train.mes.eq(month).any() or not train.mes.lt(month).any() for month in months):
        raise ValueError("Each validation month needs training and validation observations")
    test_fraction = float((~test.id_cliente.isin(train.id_cliente)).mean())
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "spec.json", spec)
    outputs, folds = [], []
    start = time.perf_counter()
    for month in months:
        cutoff = train[train.mes <= month].copy()
        x = make_features(cutoff, spec.get("blocks", []))
        a, b = cutoff.mes.lt(month), cutoff.mes.eq(month)
        no_history = ~cutoff.loc[b, "id_cliente"].isin(cutoff.loc[a, "id_cliente"])
        model = build(spec, x.loc[a], protocol["threads"])
        t = time.perf_counter()
        with threadpool_limits(limits=protocol["threads"]):
            fit_model(model, spec, x.loc[a], cutoff.loc[a, "objetivo"])
            p = model.predict_proba(x.loc[b])[:, 1]
        m = metrics(cutoff.loc[b, "objetivo"], p, no_history, test_fraction)
        if m["gini"] is None:
            raise ValueError(f"Undefined development Gini in month {month}")
        m.update(month=int(month), train_rows=int(a.sum()), seconds=time.perf_counter() - t)
        folds.append(m)
        out = cutoff.loc[b, ["id_cliente", "mes", "objetivo"]].copy()
        out["prediccion"] = p
        out["sin_historial"] = no_history.astype(int).to_numpy()
        out["cutoff"] = month
        out["model"] = spec["name"]
        outputs.append(out)
        print(f"{spec['name']} month={month} gini={m['gini']:.6f} seconds={m['seconds']:.1f}", flush=True)
    pd.concat(outputs).to_csv(run / "validation.csv", index=False, float_format="%.15g")
    result = {"spec": spec, "months": months, "folds": folds,
              "mean_gini": float(np.mean([m["gini"] for m in folds])),
              "worst_gini": min(m["gini"] for m in folds), "seconds": time.perf_counter()-start,
              "provenance": source, "feature_columns": list(x.columns),
              "identity": identity, "fingerprint": fingerprint,
              "runtime": {"timestamp_lima": now(), "python": platform.python_version(), "platform": platform.platform()},
              "prediction_path": (run / "validation.csv").relative_to(ROOT).as_posix(),
              "predictions_sha256": digest(run / "validation.csv")}
    write_json(run / "result.json", result)
    return result


def leaderboard(data_dir=None, *, workspace=None, round_config=None):
    protocol_path, runs, reports = locations(workspace, None if round_config is None else round_config["name"])
    protocol = read_json(protocol_path)
    if round_config is not None:
        protocol["round_config"] = round_config
    source = context(ROOT / "data" if data_dir is None else data_dir, protocol)
    results, excluded = [], []
    for path in runs.glob("*/*/result.json" if workspace else "*/result.json"):
        result = read_json(path)
        if (compatible(result, source, protocol["development_months"])
                and cache_matches(result, result["identity"], path.parent / "validation.csv")):
            results.append(result)
        else:
            excluded.append(path.relative_to(ROOT).as_posix())
    if excluded:
        warnings.warn(f"Leaderboard excluded {len(excluded)} incompatible or damaged runs", stacklevel=2)
    rows = []
    for r in results:
        row = {"name": r["spec"]["name"], "family": r["spec"]["family"], "blocks": "+".join(r["spec"].get("blocks", [])) or "original",
               "mean_gini": r["mean_gini"], "worst_gini": r["worst_gini"], "seconds": r["seconds"],
               "fingerprint": r["fingerprint"]}
        row.update({f"gini_{m['month']}": m["gini"] for m in r["folds"]})
        rows.append(row)
    df = pd.DataFrame(rows, columns=["name", "family", "blocks", "mean_gini", "worst_gini", "seconds", "fingerprint",
                                    *[f"gini_{m}" for m in protocol["development_months"]]])
    df = df.sort_values("mean_gini", ascending=False)
    reports.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports / "leaderboard.csv", index=False)
    write_json(reports / "leaderboard_exclusions.json", {"context": source, "excluded": excluded})
    return results, df


def baseline(data_dir):
    for family in ["logistic", "spline", "hgb"]:
        evaluate({"name": f"baseline_{family}", "family": family, "blocks": [], "params": {}, "seed": 42}, data_dir)
    _, df = leaderboard(data_dir)
    markdown(ROOT / "reports/baseline.md", "# Referencias temporales\n\nAgosto/septiembre/octubre; train estrictamente anterior a cada mes. "
             "Preprocesamiento ajustado solo en entrenamiento, early stopping desactivado. Probabilidad de clase 1.\n\n"
             + df.to_string(index=False) + "\n\nPredicciones, métricas por historial y procedencia: artifacts/runs/baseline_*/.\n")
    status("Fase 2 cerrada: `baseline` ejecutado en agosto/septiembre/octubre. Resultados en reports/baseline.md y leaderboard.csv; "
           "predicciones de validación y métricas por grupos conservadas. Siguiente: comparación controlada de bloques de variables.")


def experiment(data_dir):
    protocol = read_json(ROOT / "configs/protocol.json")
    for block in protocol["feature_blocks"]:
        evaluate({"name": f"features_{block}", "family": "hgb", "blocks": [block], "params": {}, "seed": 42}, data_dir)
    results, df = leaderboard(data_dir)
    original = next(r for r in results if r["spec"]["name"] == "baseline_hgb")
    improved = [r for r in results if r["spec"]["name"].startswith("features_") and r["mean_gini"] > original["mean_gini"]]
    blocks = [r["spec"]["blocks"][0] for r in improved]
    if len(blocks) > 1:
        combined = evaluate({"name": "features_combined", "family": "hgb", "blocks": blocks, "params": {}, "seed": 42}, data_dir)
        if combined["mean_gini"] <= original["mean_gini"]:
            blocks = max(improved, key=lambda r: r["mean_gini"])["spec"]["blocks"]
    write_json(ROOT / "configs/promising_blocks.json", {"blocks": blocks, "criterion": "HGB development mean vs original; provisional for competitive evaluation"})
    _, df = leaderboard(data_dir)
    markdown(ROOT / "reports/features.md", "# Comparación controlada de bloques\n\nMismo HGB, cortes y presupuesto; se cambia únicamente el bloque. "
             "Historial usa shift antes de rolling; meses en índice calendario; orden restaurado. Campos estáticos sin tendencias artificiales.\n\n"
             + df[df.family.eq("hgb")].to_string(index=False) + f"\n\nBloques prometedores provisionales: {blocks}. "
             "Cambios pequeños requieren análisis pareado antes de interpretar mejoras como estables.\n")
    status(f"Fase 3: comparaciones controladas terminadas (`experiment`); bloques prometedores provisionales {blocks}. "
           "Pruebas de temporalidad/orden en tests/. Siguiente: CatBoost/LightGBM y Optuna acotado.")
