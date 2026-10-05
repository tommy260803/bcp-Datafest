"""Optional paired diagnostics from existing round predictions; no model fits."""
from pathlib import Path

import pandas as pd

from .boost import candidate_a_reference, predictions, protocol
from .ensemble import aligned, paired_bootstrap
from .experiments import locations
from .hgb_round import representation, validate_plan
from .identity import context, json_hash, logical_hash, text_hash
from .io import ROOT, read_json, write_json


def diagnose(data_dir):
    config = protocol()
    plan = validate_plan(read_json(ROOT / "configs/boost/hgb_round.json"))
    _, _, reports = locations("boost", plan["name"])
    path = reports / "comparison.json"
    report = read_json(path)
    if (report["context"] != context(data_dir, dict(config, round_config=plan))
            or report["plan_sha256"] != logical_hash(plan)):
        raise ValueError("The round comparison is stale; run boost-hgb before its diagnostics")
    candidate_a_reference(data_dir)
    best = max(report["configurations"], key=lambda r: r["mean_gini"])
    frame = predictions(best)
    a = pd.read_csv(ROOT / "reports/selected_development_predictions.csv")
    reference = predictions(report["references"][representation(best["spec"]["blocks"])])
    aligned(frame, a)
    aligned(frame, reference)
    result = {"purpose": "diagnostic only; no additional models or November evaluation",
              "context": report["context"], "selected_run": best["spec"], "fingerprint": best["fingerprint"],
              "comparison_logical_sha256": json_hash(path), "diagnostic_code_sha256": text_hash(Path(__file__)),
              "method": "paired customer bootstrap; exploratory intervals after selection, not adjusted for multiple comparisons",
              "vs_a": paired_bootstrap(frame, frame.prediccion.to_numpy(), a.prediccion.to_numpy(),
                                       config["bootstrap_replicates"], config["seed"]),
              "vs_same_features": paired_bootstrap(frame, frame.prediccion.to_numpy(), reference.prediccion.to_numpy(),
                                                   config["bootstrap_replicates"], config["seed"])}
    write_json(reports / "individual_uncertainty.json", result)
    print("Best HGB vs A:", result["vs_a"], flush=True)
    print("Best HGB vs same-feature reference:", result["vs_same_features"], flush=True)
    return result
