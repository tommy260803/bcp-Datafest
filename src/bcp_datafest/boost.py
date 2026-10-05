"""Bounded development experiments for Candidate B, isolated from Candidate A."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .data import load
from .ensemble import aligned, mean_gini, paired_bootstrap
from .experiments import evaluate, leaderboard
from .identity import context, json_hash, text_hash
from .io import ROOT, digest, markdown, now, read_json, write_json
from .metrics import metrics
from .variability import audit_variability

REPORTS = ROOT / "reports/boost"


def protocol():
    config = read_json(ROOT / "configs/boost/protocol.json")
    if (config["development_months"] != [202608, 202609, 202610]
            or config["final_validation_month"] != 202611 or config["test_month"] != 202612):
        raise ValueError("Candidate B must preserve the agreed monthly protocol")
    if config["dynamic_predictors"] != ["dias_ultima_interaccion"] or config["history_window_months"] != 3:
        raise ValueError("Changing the temporal definition requires an explicit feature implementation")
    return config


def candidate_a_reference(data_dir):
    frozen = read_json(ROOT / "configs/final.json")
    source = context(data_dir, protocol())
    if source["data_hashes"] != frozen["provenance"]["data_hashes"]:
        raise ValueError("Candidate A and B must use identical original CSV files")
    paths = ["configs/final.json", "configs/protocol.json", "reports/final_validation.json",
             "reports/selected_development_predictions.csv", "reports/november_predictions.csv"]
    reference = {"schema": "candidate-reference-v2", "candidate": "A",
                 "config_logical_sha256": json_hash(ROOT / "configs/final.json"),
                 "protected_file_hashes": {p: json_hash(ROOT / p) if p.endswith(".json") else text_hash(ROOT / p) for p in paths},
                 "protected_hash_schema": "canonical-json-or-normalized-text-v1",
                 "development_mean_gini": frozen["mean_gini"], "components": frozen["components"],
                 "november_was_already_observed": True}
    path = REPORTS / "candidate_a_reference.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("schema") is None:
            # Upgrade only the initial boost reference, after proving the local
            # historical bytes have not changed. Never migrate A's files.
            legacy = dict(reference)
            legacy.pop("schema")
            legacy.pop("protected_hash_schema")
            legacy["protected_file_hashes"] = {p: digest(ROOT / p) for p in paths}
            if previous != legacy:
                raise ValueError("Legacy Candidate A reference changed")
        elif previous != reference:
            raise ValueError("Candidate A reference changed since the boost experiment began")
    write_json(path, reference)
    return frozen


def predictions(result):
    path = ROOT / result["prediction_path"]
    if digest(path) != result["predictions_sha256"]:
        raise ValueError("Prediction artifact changed")
    return pd.read_csv(path)


def summary(frame, probabilities, fraction, months):
    folds = []
    for month in months:
        mask = frame.mes.eq(month).to_numpy()
        fold = metrics(frame.objetivo.to_numpy()[mask], probabilities[mask],
                       frame.sin_historial.to_numpy()[mask], fraction)
        fold["month"] = month
        folds.append(fold)
    return {"mean_gini": float(np.mean([f["gini"] for f in folds])),
            "worst_gini": min(f["gini"] for f in folds), "folds": folds}


def passes_gate(candidate, reference, config):
    deltas = [a["gini"] - b["gini"] for a, b in zip(candidate["folds"], reference["folds"])]
    return (candidate["mean_gini"] - reference["mean_gini"] >= config["competitive_min_mean_gain"]
            and sum(delta > 0 for delta in deltas) >= config["competitive_min_improved_months"]
            and min(deltas) >= -config["competitive_max_month_loss"])


def experiment(data_dir, competitive=False):
    config = protocol()
    frozen = candidate_a_reference(data_dir)
    audit = audit_variability(data_dir)
    if audit["dynamic_predictors"] != config["dynamic_predictors"]:
        raise ValueError("Measured variability differs from the predeclared temporal feature set")
    results = {}
    for label in config["initial_hgb_variants"]:
        spec = {"name": f"b_hgb_{label}", "family": "hgb", "blocks": [] if label == "original" else [label],
                "params": {}, "seed": config["seed"]}
        results[label] = evaluate(spec, data_dir, workspace="boost")
    best_block = max(config["feature_blocks"], key=lambda b: results[b]["mean_gini"])
    # Structure can duplicate observed_time in complete monthly sequences. The
    # single exploratory CatBoost comparison must test actual interaction history.
    competitive_block = max((b for b in config["feature_blocks"] if b != "temporal_structure"),
                            key=lambda b: results[b]["mean_gini"])
    gate = passes_gate(results[best_block], results["observed_time"], config)
    # The current A spec is fixed, not retuned. Reproduction uses B's source
    # identity and saves separately; A's historical predictions stay immutable.
    a_runs = []
    for component in frozen["components"]:
        spec = copy.deepcopy(component["spec"])
        spec["name"] = "a_reproduced_" + spec["name"]
        a_runs.append(evaluate(spec, data_dir, workspace="boost"))
    if competitive:
        cat = next(c for c in frozen["components"] if c["spec"]["family"] == "catboost")
        spec = copy.deepcopy(cat["spec"])
        spec.update(name=f"b_catboost_{competitive_block}", blocks=[competitive_block])
        results["catboost_temporal"] = evaluate(spec, data_dir, workspace="boost")
        cat_reference = next(r for r in a_runs if r["spec"]["family"] == "catboost")
        cat_gate = passes_gate(results["catboost_temporal"], cat_reference, config)
        if cat_gate:
            params = {"n_estimators": 350, "num_leaves": 15, "learning_rate": .04,
                      "min_child_samples": 150, "reg_lambda": 10}
            for label, blocks in [("original", []), (competitive_block, [competitive_block])]:
                spec = {"name": f"b_lightgbm_{label}", "family": "lightgbm", "blocks": blocks,
                        "params": params, "seed": config["seed"]}
                results[f"lightgbm_{label}"] = evaluate(spec, data_dir, workspace="boost")
    train, test, _, _ = load(data_dir)
    fraction = float((~test.id_cliente.isin(train.id_cliente)).mean())
    base = predictions(a_runs[0])
    pa = np.zeros(len(base))
    for component, result in zip(frozen["components"], a_runs):
        part = predictions(result)
        aligned(base, part)
        pa += component["weight"] * part.prediccion.to_numpy()
    historical = pd.read_csv(ROOT / "reports/selected_development_predictions.csv")
    aligned(base, historical)
    if not np.allclose(pa, historical.prediccion.to_numpy(), rtol=1e-8, atol=1e-10):
        print("A reproduction differs numerically from historical predictions; see reproduction diagnostics", flush=True)
    a_metrics = summary(base, pa, fraction, config["development_months"])
    winner = max((r for k, r in results.items() if k not in ("original", "observed_time")), key=lambda r: r["mean_gini"])
    pb_frame = predictions(winner)
    aligned(base, pb_frame)
    pb = pb_frame.prediccion.to_numpy()
    confidence = paired_bootstrap(base, pb, pa, config["bootstrap_replicates"], config["seed"])
    selected_p = pb
    weights = [{"spec": winner["spec"], "weight": 1.0}]
    blends = []
    # A weaker individual can still improve an ensemble through complementary
    # errors. Test one fixed replacement, not a new weight search.
    if competitive:
        dynamic_hgb = max((results[b] for b in config["feature_blocks"] if b != "temporal_structure"),
                          key=lambda r: r["mean_gini"])
        dynamic_pred = predictions(dynamic_hgb)
        aligned(base, dynamic_pred)
        blended = np.zeros(len(base))
        replacement_components = []
        for component, a_run in zip(frozen["components"], a_runs):
            replacing = component["spec"]["family"] == "hgb"
            values = dynamic_pred.prediccion.to_numpy() if replacing else predictions(a_run).prediccion.to_numpy()
            blended += component["weight"] * values
            replacement_components.append({"spec": dynamic_hgb["spec"] if replacing else component["spec"],
                                           "weight": component["weight"]})
        blend_metrics = summary(base, blended, fraction, config["development_months"])
        best_individual = pb if winner["mean_gini"] >= a_metrics["mean_gini"] else pa
        gain = blend_metrics["mean_gini"] - max(a_metrics["mean_gini"], winner["mean_gini"])
        ci = (paired_bootstrap(base, blended, best_individual, config["bootstrap_replicates"], config["seed"])
              if gain >= config["ensemble_min_gain"] else None)
        accept_blend = (ci is not None and ci["ci95"][0] > 0
                        and passes_gate(blend_metrics, a_metrics, config))
        if accept_blend:
            selected_p = blended
            weights = replacement_components
        blends.append({"strategy": "fixed A weights; replace HGB with best dynamic HGB",
                       "metrics": blend_metrics, "components": replacement_components,
                       "gain_vs_best_individual": gain, "paired_vs_best_individual": ci,
                       "accepted": accept_blend})
    b_metrics = summary(base, selected_p, fraction, config["development_months"])
    final_ci = confidence if not blends or np.array_equal(selected_p, pb) else paired_bootstrap(
        base, selected_p, pa, config["bootstrap_replicates"], config["seed"])
    promote = (passes_gate(b_metrics, a_metrics, config) and final_ci["ci95"][0] > 0)
    result = {"timestamp_lima": now(), "context": context(data_dir, config),
              "competitive_executed": competitive, "hgb_gate_passed": gate, "best_hgb_block": best_block,
              "competitive_block": competitive_block,
              "a_reproduction": a_metrics,
              "a_historical_mean_gini": frozen["mean_gini"],
              "a_reproduction_max_probability_difference": float(np.max(np.abs(pa - historical.prediccion.to_numpy()))),
              "best_temporal_run": winner["spec"], "b_development": b_metrics,
              "b_vs_a_bootstrap": final_ci, "blends": blends, "components": weights,
              "promote_b": promote, "november_evaluated_for_b": False,
              "decision": "eligible for freezing" if promote else "retain A; no November evaluation for B"}
    write_json(REPORTS / "development_comparison.json", result)
    results_all, table = leaderboard(data_dir, workspace="boost")
    write_json(REPORTS / "development_runs.json", results_all)
    markdown(REPORTS / "development_comparison.md", "# Candidato B — desarrollo\n\n"
             "Selección exclusivamente agosto–octubre. Sin tuning inicial. "
             "Una comparación CatBoost exploratoria puede ejecutarse aunque falle la puerta HGB; "
             "LightGBM solo se abre si CatBoost mejora de forma consistente. Se comprueba una sola "
             "sustitución del HGB de A por el mejor HGB dinámico, conservando los pesos de A.\n\n```text\n"
             + table.to_string(index=False) + "\n```\n\n"
             f"A reproducido: {a_metrics['mean_gini']:.6f}. Mejor bloque HGB: {best_block}; puerta: {gate}.\n\n"
             f"B: {b_metrics['mean_gini']:.6f}; peor mes {b_metrics['worst_gini']:.6f}. "
             f"Bootstrap pareado B−A: {final_ci}.\n\nDecisión: **{result['decision']}**. "
             f"\n\nCombinación fija: {blends}.\n\n"
             "Los intervalos son exploratorios y no corrigen selección múltiple. "
             "No se evalúa noviembre con ninguna variante de B desde este comando.\n")
    print(f"A={a_metrics['mean_gini']:.6f}; B={b_metrics['mean_gini']:.6f}; {result['decision']}", flush=True)
    return result
