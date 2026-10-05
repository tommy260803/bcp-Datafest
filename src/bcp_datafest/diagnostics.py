"""Descriptive diagnosis of A using saved predictions; never trains on November."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .boost import REPORTS, candidate_a_reference, protocol
from .data import CATS, NUMS, load
from .identity import context
from .io import ROOT, markdown, read_json, write_json
from .metrics import gini


def diagnose(data_dir):
    candidate_a_reference(data_dir)
    config = protocol()
    train, test, _, _ = load(data_dir)
    dev = pd.read_csv(ROOT / "reports/selected_development_predictions.csv")
    november = pd.read_csv(ROOT / "reports/november_predictions.csv")
    saved = pd.concat([dev, november], ignore_index=True)
    joined = saved.merge(train, on=["id_cliente", "mes", "objetivo"], validate="one_to_one",
                         how="left", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Stored Candidate A predictions do not match original data")
    rows, segments, numeric, categorical = [], [], [], []
    for month in [*config["development_months"], config["final_validation_month"], config["test_month"]]:
        df = test if month == config["test_month"] else train[train.mes.eq(month)]
        no_history = ~df.id_cliente.isin(train.loc[train.mes.lt(month), "id_cliente"])
        pred = joined[joined.mes.eq(month)]
        row = {"month": month, "rows": len(df), "no_history_fraction": float(no_history.mean()),
               "positive_rate": float(df.objetivo.mean()) if "objetivo" in df else None,
               "gini": gini(pred.objetivo, pred.prediccion) if len(pred) else None}
        if len(pred):
            for name, mask in [("history", pred.sin_historial.eq(0)), ("no_history", pred.sin_historial.eq(1))]:
                row[f"gini_{name}"] = gini(pred.loc[mask, "objetivo"], pred.loc[mask, "prediccion"])
            row["score_quantiles"] = {str(q): float(pred.prediccion.quantile(q)) for q in (0, .1, .5, .9, 1)}
            for col in ("banda_riesgo", "canal_adquisicion", "region"):
                for value, part in pred.groupby(col):
                    positives, negatives = int(part.objetivo.sum()), int((1 - part.objetivo).sum())
                    sufficient = positives >= 30 and negatives >= 30
                    segments.append({"month": month, "variable": col, "value": value, "rows": len(part),
                                     "positives": positives, "positive_rate": float(part.objetivo.mean()),
                                     "mean_score": float(part.prediccion.mean()), "sufficient_classes": sufficient,
                                     "gini": gini(part.objetivo, part.prediccion) if sufficient else None})
        rows.append(row)
        for col in NUMS:
            numeric.append({"month": month, "predictor": col, "mean": float(df[col].mean()),
                            "std": float(df[col].std()), "p10": float(df[col].quantile(.1)),
                            "median": float(df[col].median()), "p90": float(df[col].quantile(.9))})
        for col in CATS:
            for value, fraction in df[col].value_counts(normalize=True).items():
                categorical.append({"month": month, "predictor": col, "value": value, "fraction": float(fraction)})
    # Paired customer-cluster resampling of months, not tuning or a model search.
    rng = np.random.default_rng(config["seed"])
    codes, customers = pd.factorize(saved.id_cliente, sort=True)
    deltas = []
    masks = {month: saved.mes.eq(month).to_numpy() for month in sorted(saved.mes.unique())}
    for _ in range(config["bootstrap_replicates"]):
        weights = np.bincount(rng.integers(0, len(customers), size=len(customers)), minlength=len(customers))[codes]
        scores = {m: gini(saved.objetivo.to_numpy()[mask], saved.prediccion.to_numpy()[mask], weights[mask])
                  for m, mask in masks.items()}
        if all(v is not None for v in scores.values()):
            deltas.append(scores[config["final_validation_month"]] - np.mean([scores[m] for m in config["development_months"]]))
    result = {"context": context(data_dir, config), "candidate": "A", "source": "saved predictions only",
              "monthly": rows, "segments": segments, "numeric": numeric, "categorical": categorical,
              "november_minus_development_ci95": np.quantile(deltas, [.025, .975]).tolist() if deltas else None,
              "valid_replicates": len(deltas), "interpretation": "Descriptive; not causal, not corrected for prior model selection. No B model is evaluated."}
    write_json(REPORTS / "november_diagnosis.json", result)
    for name, records in [("monthly", rows), ("segments", segments), ("numeric", numeric), ("categorical", categorical)]:
        pd.DataFrame(records).to_csv(REPORTS / f"drift_{name}.csv", index=False)
    validation = read_json(ROOT / "reports/final_validation.json")["metrics"]
    markdown(REPORTS / "november_diagnosis.md", "# Diagnóstico descriptivo de noviembre — Candidato A\n\n"
             "Se usan únicamente predicciones guardadas. No se entrena ni evalúa B. "
             "Diciembre solo aporta composición y distribuciones observables, sin etiquetas ni scores.\n\n```text\n"
             + pd.DataFrame(rows).drop(columns="score_quantiles", errors="ignore").to_string(index=False) + "\n```\n\n"
             f"IC95% exploratorio de noviembre menos media desarrollo, agrupando por cliente: {result['november_minus_development_ci95']}.\n\n"
             f"Gini noviembre {validation['gini']:.6f}; reponderado por composición diciembre "
             f"{validation['composition_weighted_gini']:.6f}. La ponderación por historial no explica por sí sola la caída. "
             "El grupo con historial también empeora. Las tasas, segmentos, cuantiles y frecuencias completas "
             "están en drift_*.csv. Gini por segmento se reporta solo con al menos 30 ejemplos de cada clase; "
             "la restricción del rango de riesgo puede disminuir el Gini intragrupo. "
             "Este diagnóstico no distingue causalmente drift, selección previa y variación aleatoria.\n")
    print(f"A diagnosis saved; November-development CI={result['november_minus_development_ci95']}", flush=True)
    return result
