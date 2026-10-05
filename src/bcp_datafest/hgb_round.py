"""Eight predeclared HGB configurations, then at most two fixed A replacements."""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pandas as pd

from .boost import candidate_a_reference, passes_gate, predictions, protocol, summary
from .data import load
from .ensemble import aligned, paired_bootstrap
from .experiments import evaluate, leaderboard, locations
from .identity import context, logical_hash
from .io import ROOT, digest, markdown, now, read_json, write_json


def representation(blocks):
    return "+".join(blocks) or "original"


def validate_plan(plan):
    expected = Counter({"original": 2, "observed_time": 3, "temporal_lags": 3})
    if plan["budget"] != 8 or len(plan["variants"]) != 8 or plan["max_ensembles"] != 2:
        raise ValueError("The round is limited to eight configurations and two ensembles")
    if Counter(representation(v["blocks"]) for v in plan["variants"]) != expected:
        raise ValueError("Expected two original, three observed_time and three temporal_lags configurations")
    if len({v["name"] for v in plan["variants"]}) != 8:
        raise ValueError("Configuration names must be distinct")
    if plan["fixed_params"] != {"max_iter": 150, "learning_rate": .06, "early_stopping": False}:
        raise ValueError("Iterations, learning rate and early stopping are fixed for this round")
    for variant in plan["variants"]:
        if set(variant["params"]) != {"max_leaf_nodes", "min_samples_leaf", "l2_regularization"}:
            raise ValueError("Only complexity, leaf size and regularization may vary")
        if (variant["params"]["max_leaf_nodes"] < 2 or variant["params"]["min_samples_leaf"] < 1
                or variant["params"]["l2_regularization"] < 0):
            raise ValueError("Invalid HGB parameters")
    locations("boost", plan["name"])
    return plan


def shortlist(results, references, config, limit=2):
    eligible = [r for r in results if passes_gate(r, references[representation(r["spec"]["blocks"])], config)]
    ranked = sorted(eligible, key=lambda r: (-r["mean_gini"], -r["worst_gini"], r["spec"]["name"]))
    chosen, seen = [], set()
    for result in ranked:
        label = representation(result["spec"]["blocks"])
        if label not in seen:
            chosen.append(result)
            seen.add(label)
        if len(chosen) == limit:
            break
    return chosen


def register_plan(path, plan, source):
    if path.exists():
        manifest = read_json(path)
        if manifest["plan_sha256"] != logical_hash(plan):
            raise ValueError("The registered round changed; use a new round name instead of expanding this budget")
    else:
        write_json(path, {"registered_at_lima": now(), "plan": plan,
                          "plan_sha256": logical_hash(plan), "initial_context": source})


def run(data_dir):
    config = protocol()
    plan = validate_plan(read_json(ROOT / "configs/boost/hgb_round.json"))
    frozen = candidate_a_reference(data_dir)
    _, runs_path, reports = locations("boost", plan["name"])
    round_protocol = dict(config, round_config=plan)
    source = context(data_dir, round_protocol)
    manifest_path = runs_path.parent / "manifest.json"
    # The budget is recorded before the first validation model is trained.
    register_plan(manifest_path, plan, source)
    references = {}
    for label, blocks in [("original", []), ("observed_time", ["observed_time"]), ("temporal_lags", ["temporal_lags"])]:
        spec = {"name": f"r2_reference_{label}", "family": "hgb", "blocks": blocks,
                "params": {}, "seed": config["seed"]}
        references[label] = evaluate(spec, data_dir, workspace="boost", round_config=plan)
    results = []
    for variant in plan["variants"]:
        spec = {"name": f"r2_hgb_{variant['name']}", "family": "hgb", "blocks": variant["blocks"],
                "params": dict(plan["fixed_params"], **variant["params"]), "seed": config["seed"]}
        results.append(evaluate(spec, data_dir, workspace="boost", round_config=plan))
    a_runs = []
    for component in frozen["components"]:
        spec = copy.deepcopy(component["spec"])
        if (spec["family"] == "hgb" and spec["blocks"] == ["observed_time"]
                and not spec["params"] and spec["seed"] == config["seed"]):
            result = references["observed_time"]
        else:
            spec["name"] = "r2_a_" + spec["name"]
            result = evaluate(spec, data_dir, workspace="boost", round_config=plan)
        a_runs.append(result)
    if sum(c["spec"]["family"] == "hgb" for c in frozen["components"]) != 1:
        raise ValueError("The fixed replacement requires exactly one HGB component in A")
    if (any(r["identity"]["context"] != source for r in [*results, *references.values(), *a_runs])
            or context(data_dir, round_protocol) != source):
        raise ValueError("Code, data or environment changed during the round; resume before comparing")
    train, test, _, _ = load(data_dir)
    fraction = float((~test.id_cliente.isin(train.id_cliente)).mean())
    base = predictions(a_runs[0])
    a_parts = []
    pa = np.zeros(len(base))
    for component, result in zip(frozen["components"], a_runs):
        part = predictions(result)
        aligned(base, part)
        a_parts.append(part.prediccion.to_numpy())
        pa += component["weight"] * a_parts[-1]
    historical = pd.read_csv(ROOT / "reports/selected_development_predictions.csv")
    aligned(base, historical)
    a_metrics = summary(base, pa, fraction, config["development_months"])
    selected = shortlist(results, references, config, plan["max_ensembles"])
    blends = []
    for result in selected:
        frame = predictions(result)
        aligned(base, frame)
        p = np.zeros(len(base))
        components = []
        for component, a_part in zip(frozen["components"], a_parts):
            replacing = component["spec"]["family"] == "hgb"
            p += component["weight"] * (frame.prediccion.to_numpy() if replacing else a_part)
            components.append({"spec": result["spec"] if replacing else component["spec"], "weight": component["weight"]})
        scores = summary(base, p, fraction, config["development_months"])
        gain = scores["mean_gini"] - max(a_metrics["mean_gini"], result["mean_gini"])
        stable = passes_gate(scores, a_metrics, config)
        ci = None
        if gain >= config["ensemble_min_gain"] and stable:
            reference = pa if a_metrics["mean_gini"] >= result["mean_gini"] else frame.prediccion.to_numpy()
            ci = paired_bootstrap(base, p, reference, config["bootstrap_replicates"], config["seed"])
        accepted = ci is not None and ci["ci95"][0] > 0
        reason = ("eligible for freezing" if accepted else
                  "gain below the predeclared minimum" if gain < config["ensemble_min_gain"] else
                  "development stability gate failed" if not stable else "bootstrap lower bound is not positive")
        blend = {"replacement": result["spec"], "components": components, "metrics": scores,
                 "gain_vs_a": scores["mean_gini"] - a_metrics["mean_gini"],
                 "gain_vs_best_individual": gain, "stability_passed": stable,
                 "paired_bootstrap": ci, "accepted": accepted, "reason": reason}
        output = base.copy()
        output["prediccion"] = p
        output["model"] = "ensemble_" + result["spec"]["name"]
        fingerprint = logical_hash({"context": source, "components": components})
        output_path = runs_path.parent / "ensembles" / fingerprint / "validation.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, float_format="%.15g")
        blend.update(fingerprint=fingerprint, prediction_path=output_path.relative_to(ROOT).as_posix(),
                     predictions_sha256=digest(output_path))
        blends.append(blend)
    accepted = [b for b in blends if b["accepted"]]
    winner = max(accepted, key=lambda b: b["metrics"]["mean_gini"]) if accepted else None
    report = {"timestamp_lima": now(), "context": source, "plan": plan,
              "plan_sha256": logical_hash(plan), "budget_consumed": len(results),
              "references": references, "configurations": results,
              "shortlist": [r["spec"]["name"] for r in selected], "ensembles": blends,
              "a_reproduction": a_metrics,
              "a_max_probability_difference": float(np.max(np.abs(pa - historical.prediccion.to_numpy()))),
              "promote_b": winner is not None, "selection": winner,
              "november_evaluated_for_b": False,
              "decision": "eligible for freezing" if winner else "retain A; stop this HGB round"}
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "comparison.json", report)
    write_json(runs_path.parent / "comparisons" / logical_hash(source) / "comparison.json", report)
    _, table = leaderboard(data_dir, workspace="boost", round_config=plan)
    candidate_rows = []
    for result in results:
        reference = references[representation(result["spec"]["blocks"])]
        row = {"name": result["spec"]["name"], "representation": representation(result["spec"]["blocks"]),
               "mean_gini": result["mean_gini"], "worst_gini": result["worst_gini"],
               "gain_vs_same_features": result["mean_gini"] - reference["mean_gini"],
               "shortlist_gate_passed": passes_gate(result, reference, config)}
        row.update({f"gini_{f['month']}": f["gini"] for f in result["folds"]})
        candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows).sort_values("mean_gini", ascending=False)
    candidates.to_csv(reports / "hgb_comparison.csv", index=False)
    markdown(reports / "comparison.md", "# Ronda 2 — HGB acotado\n\n"
             f"Ocho configuraciones predeclaradas; hasta dos reemplazos con pesos de A intactos. "
             f"Plan lógico: {report['plan_sha256']}.\n\n```text\n"
             + candidates.to_string(index=False) + "\n```\n\n"
             f"A: {a_metrics['mean_gini']:.6f}; diferencia máxima frente a predicciones históricas: "
             f"{report['a_max_probability_difference']:.3g}.\n\nShortlist: {report['shortlist']}.\n\n"
             + "\n".join(f"- {b['replacement']['name']}: media {b['metrics']['mean_gini']:.6f}; "
                         f"ganancia A {b['gain_vs_a']:+.6f}; peor mes {b['metrics']['worst_gini']:.6f}; {b['reason']}." for b in blends)
             + f"\n\n**Decisión: {report['decision']}.** Noviembre no evaluado para B. "
             "Los intervalos, si se calculan, son exploratorios sin corrección por selección múltiple. "
             "Leaderboard y métricas por historial completos en los archivos de esta ronda.\n")
    print(candidates.to_string(index=False), flush=True)
    print(f"Shortlist={report['shortlist']}; {report['decision']}", flush=True)
    return report
