"""Six controlled CatBoost candidates and at most two fixed-weight ensembles."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .boost import candidate_a_reference, passes_gate, predictions, protocol, summary
from .data import load
from .ensemble import aligned, paired_bootstrap
from .experiments import evaluate, leaderboard, locations
from .hgb_round import register_plan
from .identity import context, logical_hash
from .io import ROOT, digest, markdown, now, read_json, write_json


def validate_plan(plan):
    if plan["budget"] != 6 or len(plan["variants"]) != 6 or plan["max_ensembles"] != 2:
        raise ValueError("CatBoost round is limited to six configurations and two ensembles")
    if plan["blocks"]:
        raise ValueError("This round uses only original predictors")
    if len({v["name"] for v in plan["variants"]}) != 6:
        raise ValueError("CatBoost configuration names must be distinct")
    allowed = {"depth", "l2_leaf_reg", "iterations", "learning_rate"}
    for variant in plan["variants"]:
        overrides = variant["overrides"]
        if not overrides or not set(overrides).issubset(allowed):
            raise ValueError("Only complexity, L2, iterations and learning rate may vary")
        if (not 2 <= overrides.get("depth", 3) <= 4
                or overrides.get("l2_leaf_reg", 1) <= 0
                or not 300 <= overrides.get("iterations", 600) <= 900
                or not 0 < overrides.get("learning_rate", .04) <= .1):
            raise ValueError("CatBoost overrides exceed the bounded search")
    alt = plan["hgb_alternative"]
    if (alt["family"] != "hgb" or alt["blocks"] != ["observed_time"] or alt["seed"] != 42
            or alt["params"] != {"max_iter": 150, "learning_rate": .06, "early_stopping": False,
                                 "max_leaf_nodes": 7, "min_samples_leaf": 200, "l2_regularization": 30}):
        raise ValueError("The HGB alternative must be the preselected round-2 shallow model")
    locations("boost", plan["name"])
    return plan


def select_catboost(results, reference, config):
    eligible = [r for r in results if passes_gate(r, reference, config)]
    return min(eligible, key=lambda r: (-r["mean_gini"], -r["worst_gini"], r["spec"]["name"])) if eligible else None


def save_prediction(base, p, components, source, directory, name):
    fingerprint = logical_hash({"context": source, "components": components})
    path = directory / fingerprint / "validation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = base.copy()
    frame["prediccion"] = p
    frame["model"] = name
    frame.to_csv(path, index=False, float_format="%.15g")
    return {"fingerprint": fingerprint, "prediction_path": path.relative_to(ROOT).as_posix(),
            "predictions_sha256": digest(path)}


def run(data_dir):
    config = protocol()
    plan = validate_plan(read_json(ROOT / "configs/boost/catboost_round.json"))
    frozen = candidate_a_reference(data_dir)
    if ([c["spec"]["family"] for c in frozen["components"]] != ["catboost", "hgb"]
            or [c["weight"] for c in frozen["components"]] != [.75, .25]
            or frozen["components"][0]["spec"]["blocks"]):
        raise ValueError("This round is anchored to A's original CatBoost/HGB 75/25 components")
    _, runs_path, reports = locations("boost", plan["name"])
    round_protocol = dict(config, round_config=plan)
    source = context(data_dir, round_protocol)
    register_plan(runs_path.parent / "manifest.json", plan, source)
    a_runs = []
    for component in frozen["components"]:
        spec = copy.deepcopy(component["spec"])
        spec["name"] = "r3_a_" + spec["name"]
        a_runs.append(evaluate(spec, data_dir, workspace="boost", round_config=plan))
    hgb_alt = evaluate(plan["hgb_alternative"], data_dir, workspace="boost", round_config=plan)
    anchor = frozen["components"][0]["spec"]
    results = []
    for variant in plan["variants"]:
        spec = copy.deepcopy(anchor)
        spec["name"] = "r3_catboost_" + variant["name"]
        spec["params"].update(variant["overrides"])
        results.append(evaluate(spec, data_dir, workspace="boost", round_config=plan))
    if (any(r["identity"]["context"] != source for r in [*results, *a_runs, hgb_alt])
            or context(data_dir, round_protocol) != source):
        raise ValueError("Code, data or environment changed during the round; resume before comparing")
    train, test, _, _ = load(data_dir)
    fraction = float((~test.id_cliente.isin(train.id_cliente)).mean())
    base = predictions(a_runs[0])
    hgb_a = predictions(a_runs[1])
    shallow = predictions(hgb_alt)
    aligned(base, hgb_a)
    aligned(base, shallow)
    pa = .75 * base.prediccion.to_numpy() + .25 * hgb_a.prediccion.to_numpy()
    historical = pd.read_csv(ROOT / "reports/selected_development_predictions.csv")
    aligned(base, historical)
    a_metrics = summary(base, pa, fraction, config["development_months"])
    chosen = select_catboost(results, a_runs[0], config)
    single, blends, accepted = None, [], []
    if chosen is not None:
        frame = predictions(chosen)
        aligned(base, frame)
        pc = frame.prediccion.to_numpy()
        single_gate = passes_gate(chosen, a_metrics, config)
        ci = paired_bootstrap(base, pc, pa, config["bootstrap_replicates"], config["seed"]) if single_gate else None
        single = {"spec": chosen["spec"], "metrics": {k: chosen[k] for k in ("mean_gini", "worst_gini", "folds")},
                  "components": [{"spec": chosen["spec"], "weight": 1.0}],
                  "gain_vs_a": chosen["mean_gini"] - a_metrics["mean_gini"], "stability_passed": single_gate,
                  "paired_vs_a": ci, "accepted": ci is not None and ci["ci95"][0] > 0,
                  "prediction_path": chosen["prediction_path"], "predictions_sha256": chosen["predictions_sha256"],
                  "fingerprint": chosen["fingerprint"]}
        if single["accepted"]:
            accepted.append(single)
        for hgb_result, hgb_frame in [(a_runs[1], hgb_a), (hgb_alt, shallow)]:
            p = .75 * pc + .25 * hgb_frame.prediccion.to_numpy()
            scores = summary(base, p, fraction, config["development_months"])
            best_score = max(a_metrics["mean_gini"], chosen["mean_gini"], hgb_result["mean_gini"])
            gain = scores["mean_gini"] - best_score
            stable = passes_gate(scores, a_metrics, config)
            vs_a, vs_individual = None, None
            if gain >= config["ensemble_min_gain"] and stable:
                strongest = pc if chosen["mean_gini"] >= hgb_result["mean_gini"] else hgb_frame.prediccion.to_numpy()
                if a_metrics["mean_gini"] > max(chosen["mean_gini"], hgb_result["mean_gini"]):
                    strongest = pa
                vs_a = paired_bootstrap(base, p, pa, config["bootstrap_replicates"], config["seed"])
                vs_individual = (vs_a if strongest is pa else
                                 paired_bootstrap(base, p, strongest, config["bootstrap_replicates"], config["seed"]))
            ok = vs_a is not None and vs_a["ci95"][0] > 0 and vs_individual["ci95"][0] > 0
            components = [{"spec": chosen["spec"], "weight": .75}, {"spec": hgb_result["spec"], "weight": .25}]
            reason = ("eligible for freezing" if ok else "gain below the predeclared minimum"
                      if gain < config["ensemble_min_gain"] else "development stability gate failed"
                      if not stable else "bootstrap lower bound is not positive")
            blend = {"components": components, "metrics": scores,
                     "gain_vs_a": scores["mean_gini"] - a_metrics["mean_gini"],
                     "gain_vs_best_individual": gain, "stability_passed": stable,
                     "paired_vs_a": vs_a, "paired_vs_best_individual": vs_individual, "accepted": ok, "reason": reason}
            blend.update(save_prediction(base, p, components, source, runs_path.parent / "ensembles",
                                         "ensemble_" + hgb_result["spec"]["name"]))
            blends.append(blend)
            if ok:
                accepted.append(blend)
    winner = max(accepted, key=lambda r: r["metrics"]["mean_gini"]) if accepted else None
    report = {"timestamp_lima": now(), "context": source, "plan": plan, "plan_sha256": logical_hash(plan),
              "budget_consumed": len(results), "configurations": results, "a_runs": a_runs,
              "hgb_alternative": hgb_alt, "a_reproduction": a_metrics,
              "a_max_probability_difference": float(np.max(np.abs(pa - historical.prediccion.to_numpy()))),
              "shortlist": chosen["spec"] if chosen else None, "single_candidate": single,
              "ensembles": blends, "promote_b": winner is not None, "selection": winner,
              "november_evaluated_for_b": False,
              "decision": "eligible for freezing" if winner else "retain A; stop this CatBoost round"}
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "comparison.json", report)
    write_json(runs_path.parent / "comparisons" / logical_hash(source) / "comparison.json", report)
    leaderboard(data_dir, workspace="boost", round_config=plan)
    rows = []
    for result in results:
        row = {"name": result["spec"]["name"], "mean_gini": result["mean_gini"], "worst_gini": result["worst_gini"],
               "gain_vs_catboost_a": result["mean_gini"] - a_runs[0]["mean_gini"],
               "shortlist_gate_passed": passes_gate(result, a_runs[0], config)}
        row.update({f"gini_{f['month']}": f["gini"] for f in result["folds"]})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("mean_gini", ascending=False)
    table.to_csv(reports / "catboost_comparison.csv", index=False)
    markdown(reports / "comparison.md", "# Ronda 3 — CatBoost original acotado\n\n"
             f"Seis configuraciones predeclaradas; hasta dos combinaciones 75/25. Plan: {report['plan_sha256']}.\n\n```text\n"
             + table.to_string(index=False) + "\n```\n\n"
             f"A reproducido {a_metrics['mean_gini']:.6f}; diferencia máxima {report['a_max_probability_difference']:.3g}.\n\n"
             f"Shortlist: {report['shortlist']}.\n\nIndividual: {single}.\n\n"
             + "\n".join(f"- HGB {b['components'][1]['spec']['name']}: media {b['metrics']['mean_gini']:.6f}, "
                         f"delta A {b['gain_vs_a']:+.6f}, peor mes {b['metrics']['worst_gini']:.6f}; {b['reason']}." for b in blends)
             + f"\n\n**Decisión: {report['decision']}.** Noviembre no evaluado para B. "
             "Los intervalos, cuando se calculan, son exploratorios y no corrigen la selección acumulada de candidatos.\n")
    print(table.to_string(index=False), flush=True)
    print(report["decision"], flush=True)
    return report
