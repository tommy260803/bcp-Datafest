import copy

import numpy as np
import pytest

from bcp_datafest import experiments
from bcp_datafest.hgb_round import register_plan, shortlist, validate_plan
from bcp_datafest.io import ROOT, read_json
from test_experiment_identity import install_fake_experiment


def test_plan_is_bounded_and_cannot_expand_after_registration(tmp_path):
    plan = validate_plan(read_json(ROOT / "configs/boost/hgb_round.json"))
    path = tmp_path / "manifest.json"
    register_plan(path, plan, {"code": "first"})
    register_plan(path, plan, {"code": "second"})
    assert read_json(path)["initial_context"] == {"code": "first"}
    changed = copy.deepcopy(plan)
    changed["variants"][0]["params"]["l2_regularization"] = 40
    with pytest.raises(ValueError, match="registered round changed"):
        register_plan(path, changed, {})
    changed["fixed_params"]["learning_rate"] = .1
    with pytest.raises(ValueError, match="fixed"):
        validate_plan(changed)


def result(name, blocks, scores):
    return {"spec": {"name": name, "blocks": blocks}, "mean_gini": float(np.mean(scores)),
            "worst_gini": min(scores), "folds": [{"month": m, "gini": score}
                                                for m, score in zip([202608, 202609, 202610], scores)]}


def test_shortlist_limits_comparisons_and_rejects_single_month_overfit():
    config = {"competitive_min_mean_gain": .0005, "competitive_min_improved_months": 2,
              "competitive_max_month_loss": .002}
    references = {key: result(key, blocks, [.25] * 3) for key, blocks in
                  [("original", []), ("observed_time", ["observed_time"]), ("temporal_lags", ["temporal_lags"])]}
    cases = [result("obs_best", ["observed_time"], [.255, .254, .253]),
             result("obs_second", ["observed_time"], [.252, .253, .252]),
             result("original", [], [.253, .252, .251]),
             result("lags_overfit", ["temporal_lags"], [.240, .275, .275])]
    chosen = shortlist(cases, references, config)
    assert [r["spec"]["name"] for r in chosen] == ["obs_best", "original"]


def test_round_cache_and_reports_are_isolated(monkeypatch, tmp_path):
    source, fits = install_fake_experiment(monkeypatch, tmp_path)
    monkeypatch.setattr(experiments, "context", lambda data, protocol: dict(source, protocol=protocol))
    spec = {"name": "same", "family": "hgb", "blocks": [], "params": {}}
    first_round = experiments.evaluate(spec, tmp_path, workspace="boost")
    sentinel = tmp_path / "reports/boost/leaderboard.csv"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("first round intact", encoding="utf-8")
    plan = {"name": "hgb_round_2", "budget": 8}
    second_round = experiments.evaluate(spec, tmp_path, workspace="boost", round_config=plan)
    assert second_round["prediction_path"] != first_round["prediction_path"]
    assert len(fits) == 2
    assert experiments.evaluate(spec, tmp_path, workspace="boost", round_config=plan) == second_round
    assert len(fits) == 2
    results, _ = experiments.leaderboard(tmp_path, workspace="boost", round_config=plan)
    assert len(results) == 1
    assert sentinel.read_text(encoding="utf-8") == "first round intact"
    assert (tmp_path / "reports/boost/rounds/hgb_round_2/leaderboard.csv").is_file()
    changed = dict(plan, budget=9)
    experiments.evaluate(spec, tmp_path, workspace="boost", round_config=changed)
    assert len(fits) == 3  # the round plan itself participates in cache identity
