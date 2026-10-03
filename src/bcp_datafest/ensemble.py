from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load
from .experiments import leaderboard
from .io import ROOT, read_json, write_json, markdown, status, now, digest, provenance
from .metrics import gini, metrics


def read_predictions(name):
    return pd.read_csv(ROOT / "artifacts/runs" / name / "validation.csv")


def aligned(a, b):
    if not a[["id_cliente", "mes", "objetivo", "sin_historial"]].equals(b[["id_cliente", "mes", "objetivo", "sin_historial"]]):
        raise ValueError("Validation predictions not aligned")


def mean_gini(df, p, weights=None):
    vals = []
    for month in sorted(df.mes.unique()):
        mask = df.mes.eq(month).to_numpy()
        score = gini(df.objetivo.to_numpy()[mask], p[mask], None if weights is None else weights[mask])
        if score is None:
            return None
        vals.append(score)
    return float(np.mean(vals))


def paired_bootstrap(df, p, reference, replicates=500):
    """Resample customers jointly across months, retain identical paired rows."""
    rng = np.random.default_rng(42)
    codes, customers = pd.factorize(df.id_cliente, sort=True)
    n = len(customers)
    differences = []
    for _ in range(replicates):
        multiplicities = np.bincount(rng.integers(0, n, size=n), minlength=n)
        weights = multiplicities[codes]
        # Each bootstrap sample has both classes for these data; use weights
        # so repeat observations remain in the same resampled customer cluster.
        differences.append(mean_gini(df, p, weights) - mean_gini(df, reference, weights))
    return {"replicates": replicates, "unit": "customer (jointly across months)",
            "mean_delta": mean_gini(df, p)-mean_gini(df, reference),
            "ci95": [float(v) for v in np.quantile(differences, [.025, .975])]}


def select(data_dir):
    frozen = ROOT / "configs/final.json"
    if frozen.exists():
        print("Selection already frozen; refusing to change it. See configs/final.json", flush=True)
        return read_json(frozen)
    protocol = read_json(ROOT / "configs/protocol.json")
    for family in ["catboost", "lightgbm"]:
        study = read_json(ROOT / f"reports/{family}_study.json")
        assert study["complete_trials"] >= protocol["tuning_trials_per_family"], "Tuning budget incomplete"
    results, df = leaderboard()
    by_name = {r["spec"]["name"]: r for r in results}
    winner = df.iloc[0]["name"]
    base = read_predictions(winner)
    pbest = base.prediccion.to_numpy()
    specs = [{"spec": by_name[winner]["spec"], "weight": 1.0}]
    # Fixed small grid, two best distinct model families, same OOT rows.
    strongest = df.drop_duplicates("family").head(2)["name"].tolist()
    blend_rows = []
    candidate_p = None
    candidate_weight = None
    if len(strongest) == 2:
        other = read_predictions(strongest[1])
        aligned(base, other)
        for w in [.25, .5, .75]:
            p = w*pbest + (1-w)*other.prediccion.to_numpy()
            blend_rows.append({"first": winner, "second": strongest[1], "first_weight": w,
                               "mean_gini": mean_gini(base, p)})
        chosen = max(blend_rows, key=lambda r: r["mean_gini"])
        candidate_weight = chosen["first_weight"]
        candidate_p = candidate_weight*pbest + (1-candidate_weight)*other.prediccion.to_numpy()
    ci = paired_bootstrap(base, candidate_p, pbest, protocol["bootstrap_replicates"]) if candidate_p is not None else None
    accept = ci is not None and ci["mean_delta"] >= protocol["ensemble_min_gain"] and ci["ci95"][0] > 0
    if accept:
        specs = [{"spec": by_name[winner]["spec"], "weight": candidate_weight},
                 {"spec": by_name[strongest[1]]["spec"], "weight": 1-candidate_weight}]
        selected_p = candidate_p
    else:
        selected_p = pbest
    runnerup = df.iloc[1]["name"]
    second = read_predictions(runnerup)
    aligned(base, second)
    individual_ci = paired_bootstrap(base, pbest, second.prediccion.to_numpy(), protocol["bootstrap_replicates"])
    train, test, _, _ = load(data_dir)
    test_fraction = float((~test.id_cliente.isin(train.id_cliente)).mean())
    folds = []
    for month in protocol["development_months"]:
        mask = base.mes.eq(month).to_numpy()
        m = metrics(base.objetivo.to_numpy()[mask], selected_p[mask], base.sin_historial.to_numpy()[mask], test_fraction)
        m["month"] = month
        folds.append(m)
    result = {"frozen_at_lima": now(), "components": specs, "development_months": protocol["development_months"],
              "iterations_policy": protocol["iterations_policy"], "mean_gini": mean_gini(base, selected_p),
              "folds": folds, "selection_metric": protocol["selection_metric"], "ensemble_accepted": bool(accept),
              "ensemble_vs_best_individual": ci, "best_vs_runnerup": individual_ci,
              "feature_columns": {c["spec"]["name"]: by_name[c["spec"]["name"]]["feature_columns"] for c in specs},
              "test_no_history_fraction": test_fraction, "provenance": provenance(data_dir)}
    write_json(frozen, result)
    write_json(ROOT / "reports/selection.json", result)
    write_json(ROOT / "reports/blend_candidates.json", blend_rows)
    base["prediccion"] = selected_p
    base.to_csv(ROOT / "reports/selected_development_predictions.csv", index=False, float_format="%.15g")
    markdown(ROOT / "reports/model_selection.md", "# Selección congelada antes de noviembre\n\n"
             f"Fecha Lima: {result['frozen_at_lima']}. Configuración: configs/final.json; SHA-256: {digest(frozen)}.\n\n"
             f"Componentes elegidos: {specs}. Media mensual Gini: {result['mean_gini']:.6f}.\n\n"
             "Clasificación de candidatos:\n\n" + df.to_string(index=False) +
             f"\n\nCombinaciones probadas: {blend_rows}.\n\nBootstrap pareado por cliente de mejor combinación frente a mejor individual: {ci}. "
             f"Aceptación predefinida: {protocol['ensemble_acceptance']}. Ensamble aceptado: {accept}.\n\n"
             f"Mejor individual frente al segundo: {individual_ci}. Intervalos exploratorios posteriores a búsqueda: "
             "no corrigen el optimismo por comparar múltiples candidatos y no garantizan superioridad fuera de desarrollo.\n\n"
             f"Composición diciembre sin historial: {test_fraction:.6%}; diagnóstico ponderado y por grupos en selection.json. "
             "Decisión usa exclusivamente agosto/septiembre/octubre. Iteraciones fijas, sin calibración ni balanceo.\n")
    status(f"Fase 5 cerrada: configuración congelada en configs/final.json (SHA-256 {digest(frozen)}). "
           f"Media Gini desarrollo {result['mean_gini']:.6f}, componentes {specs}. Bootstrap y combinaciones registrados; "
           "siguiente: única evaluación de noviembre con configuración congelada.")
    print(result, flush=True)
    return result
