from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load
from .features import make_features
from .io import ROOT, read_json, write_json, provenance, markdown, status
from .metrics import metrics
from .models import build, fit_model


def evaluate(spec, data_dir, months=None):
    protocol = read_json(ROOT / "configs/protocol.json")
    months = months or protocol["development_months"]
    train, test, _, _ = load(data_dir)
    source = provenance(data_dir)
    test_fraction = float((~test.id_cliente.isin(train.id_cliente)).mean())
    run = ROOT / "artifacts/runs" / spec["name"]
    if (run / "result.json").exists():
        previous = read_json(run / "result.json")
        assert previous["spec"] == spec and previous["months"] == months, "Run ID collision"
        assert previous["provenance"]["data_hashes"] == provenance(data_dir)["data_hashes"], "Data changed"
        return previous
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
              "provenance": source, "feature_columns": list(x.columns)}
    write_json(run / "result.json", result)
    return result


def leaderboard():
    results = [read_json(p) for p in (ROOT / "artifacts/runs").glob("*/result.json")]
    results = [r for r in results if r["months"] == read_json(ROOT / "configs/protocol.json")["development_months"]]
    rows = []
    for r in results:
        row = {"name": r["spec"]["name"], "family": r["spec"]["family"], "blocks": "+".join(r["spec"].get("blocks", [])) or "original",
               "mean_gini": r["mean_gini"], "worst_gini": r["worst_gini"], "seconds": r["seconds"]}
        row.update({f"gini_{m['month']}": m["gini"] for m in r["folds"]})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mean_gini", ascending=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports/leaderboard.csv", index=False)
    return results, df


def baseline(data_dir):
    for family in ["logistic", "spline", "hgb"]:
        evaluate({"name": f"baseline_{family}", "family": family, "blocks": [], "params": {}, "seed": 42}, data_dir)
    _, df = leaderboard()
    markdown(ROOT / "reports/baseline.md", "# Referencias temporales\n\nAgosto/septiembre/octubre; train estrictamente anterior a cada mes. "
             "Preprocesamiento ajustado solo en entrenamiento, early stopping desactivado. Probabilidad de clase 1.\n\n"
             + df.to_string(index=False) + "\n\nPredicciones, métricas por historial y procedencia: artifacts/runs/baseline_*/.\n")
    status("Fase 2 cerrada: `baseline` ejecutado en agosto/septiembre/octubre. Resultados en reports/baseline.md y leaderboard.csv; "
           "predicciones de validación y métricas por grupos conservadas. Siguiente: comparación controlada de bloques de variables.")


def experiment(data_dir):
    protocol = read_json(ROOT / "configs/protocol.json")
    for block in protocol["feature_blocks"]:
        evaluate({"name": f"features_{block}", "family": "hgb", "blocks": [block], "params": {}, "seed": 42}, data_dir)
    results, df = leaderboard()
    original = next(r for r in results if r["spec"]["name"] == "baseline_hgb")
    improved = [r for r in results if r["spec"]["name"].startswith("features_") and r["mean_gini"] > original["mean_gini"]]
    blocks = [r["spec"]["blocks"][0] for r in improved]
    if len(blocks) > 1:
        combined = evaluate({"name": "features_combined", "family": "hgb", "blocks": blocks, "params": {}, "seed": 42}, data_dir)
        if combined["mean_gini"] <= original["mean_gini"]:
            blocks = max(improved, key=lambda r: r["mean_gini"])["spec"]["blocks"]
    write_json(ROOT / "configs/promising_blocks.json", {"blocks": blocks, "criterion": "HGB development mean vs original; provisional for competitive evaluation"})
    _, df = leaderboard()
    markdown(ROOT / "reports/features.md", "# Comparación controlada de bloques\n\nMismo HGB, cortes y presupuesto; se cambia únicamente el bloque. "
             "Historial usa shift antes de rolling; meses en índice calendario; orden restaurado. Campos estáticos sin tendencias artificiales.\n\n"
             + df[df.family.eq("hgb")].to_string(index=False) + f"\n\nBloques prometedores provisionales: {blocks}. "
             "Cambios pequeños requieren análisis pareado antes de interpretar mejoras como estables.\n")
    status(f"Fase 3: comparaciones controladas terminadas (`experiment`); bloques prometedores provisionales {blocks}. "
           "Pruebas de temporalidad/orden en tests/. Siguiente: CatBoost/LightGBM y Optuna acotado.")
