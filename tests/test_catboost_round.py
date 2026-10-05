import copy

import numpy as np
import pytest

from bcp_datafest import catboost_round
from bcp_datafest.io import ROOT, read_json, write_json


def result(name, scores):
    return {"spec": {"name": name}, "mean_gini": float(np.mean(scores)), "worst_gini": min(scores),
            "folds": [{"month": m, "gini": score} for m, score in zip([202608, 202609, 202610], scores)]}


def test_shortlist_rejects_unstable_high_mean_and_can_stop():
    config = {"competitive_min_mean_gain": .0005, "competitive_min_improved_months": 2,
              "competitive_max_month_loss": .002}
    baseline = result("baseline", [.25] * 3)
    unstable = result("unstable", [.24, .27, .27])
    stable = result("stable", [.255, .253, .252])
    assert catboost_round.select_catboost([unstable, stable], baseline, config) == stable
    assert catboost_round.select_catboost([unstable], baseline, config) is None


def test_plan_prevents_feature_seed_or_budget_expansion():
    plan = read_json(ROOT / "configs/boost/catboost_round.json")
    catboost_round.validate_plan(plan)
    for change in ("features", "seed", "budget"):
        changed = copy.deepcopy(plan)
        if change == "features":
            changed["blocks"] = ["temporal_lags"]
        elif change == "seed":
            changed["variants"][0]["overrides"]["random_seed"] = 99
        else:
            changed["budget"] = 7
        with pytest.raises(ValueError):
            catboost_round.validate_plan(changed)


def test_round_preserves_anchor_and_refuses_cross_version_comparison(monkeypatch, tmp_path):
    plan = read_json(ROOT / "configs/boost/catboost_round.json")
    write_json(tmp_path / "configs/boost/catboost_round.json", plan)
    monkeypatch.setattr(catboost_round, "ROOT", tmp_path)
    monkeypatch.setattr(catboost_round, "protocol", lambda: {})
    frozen = {"components": [
        {"weight": .75, "spec": {"name": "cat", "family": "catboost", "blocks": [], "seed": 42,
                                  "params": {"depth": 3, "iterations": 600, "l2_leaf_reg": 31.6, "learning_rate": .04}}},
        {"weight": .25, "spec": {"name": "hgb", "family": "hgb", "blocks": ["observed_time"], "seed": 42, "params": {}}}
    ]}
    before = copy.deepcopy(frozen)
    monkeypatch.setattr(catboost_round, "candidate_a_reference", lambda _: frozen)
    monkeypatch.setattr(catboost_round, "context", lambda *args: {"code": "current"})
    monkeypatch.setattr(catboost_round, "locations", lambda *args: (
        tmp_path / "protocol.json", tmp_path / "artifacts/round/runs", tmp_path / "reports/round"))
    evaluated = []

    def fake_evaluate(spec, *args, **kwargs):
        evaluated.append(copy.deepcopy(spec))
        return {"identity": {"context": {"code": "stale"}}}

    monkeypatch.setattr(catboost_round, "evaluate", fake_evaluate)
    with pytest.raises(ValueError, match="changed during the round"):
        catboost_round.run(tmp_path)
    assert frozen == before
    assert len(evaluated) == 9  # six candidates plus A's two components and fixed HGB
    assert all(spec["blocks"] == [] and spec["seed"] == 42 for spec in evaluated[3:])
    assert read_json(tmp_path / "artifacts/round/manifest.json")["plan"]["budget"] == 6
