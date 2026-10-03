from __future__ import annotations

import optuna
import joblib

from .experiments import evaluate, leaderboard
from .io import ROOT, read_json, write_json, status


def suggest(trial, family):
    if family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 800, step=100),
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", .025, .1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 40, log=True),
            "random_strength": trial.suggest_float("random_strength", .1, 3, log=True),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "learning_rate": trial.suggest_float("learning_rate", .02, .08, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 500, step=50),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 40, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", .7, 1),
        "subsample": trial.suggest_float("subsample", .7, 1),
        "subsample_freq": 1,
    }


def tune(data_dir, family, trials=None):
    assert family in ["catboost", "lightgbm"]
    if (ROOT / "configs/final.json").exists():
        raise RuntimeError("Selection frozen: tuning is closed to preserve the November holdout protocol")
    protocol = read_json(ROOT / "configs/protocol.json")
    blocks = read_json(ROOT / "configs/promising_blocks.json")["blocks"]
    params = ({"iterations": 500, "depth": 4, "learning_rate": .05, "l2_leaf_reg": 10} if family == "catboost" else
              {"n_estimators": 350, "num_leaves": 15, "learning_rate": .04, "min_child_samples": 150, "reg_lambda": 10})
    for label, features in [("original", []), ("promising", blocks)]:
        if label == "promising" and not features:
            continue
        evaluate({"name": f"{family}_{label}", "family": family, "blocks": features, "params": params, "seed": 42}, data_dir)
    # Preserve the sampler RNG for ordinary resumes at completed study boundaries.
    # Interrupted optimization can leave RUNNING trials; those remain visible.
    directory = ROOT / "artifacts/studies"
    directory.mkdir(parents=True, exist_ok=True)
    sampler_path = directory / f"{family}_sampler.joblib"
    sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=42, n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                storage=f"sqlite:///{(directory / (family + '.sqlite3')).as_posix()}",
                                study_name=family, load_if_exists=True)
    n = trials if trials is not None else protocol["tuning_trials_per_family"]

    def objective(trial):
        chosen = trial.suggest_categorical("feature_set", ["original", "promising"] if blocks else ["original"])
        p = suggest(trial, family)
        spec = {"name": f"{family}_trial_{trial.number:03d}", "family": family,
                "blocks": blocks if chosen == "promising" else [], "params": p, "seed": 42}
        try:
            result = evaluate(spec, data_dir)
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            write_json(directory / f"{family}_trial_{trial.number:03d}_failure.json", {"spec": spec, "error": repr(exc)})
            raise
        trial.set_user_attr("run_id", spec["name"])
        trial.set_user_attr("seconds", result["seconds"])
        trial.set_user_attr("fold_gini", [r["gini"] for r in result["folds"]])
        return result["mean_gini"]

    remaining = max(0, n - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, catch=(ValueError, RuntimeError))
    joblib.dump(study.sampler, sampler_path)
    study.trials_dataframe().to_csv(ROOT / f"reports/{family}_trials.csv", index=False)
    write_json(ROOT / f"reports/{family}_study.json", {"best_value": study.best_value, "best_params": study.best_params,
               "trials": len(study.trials), "complete_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
               "failed_trials": [{"number": t.number, "state": t.state.name} for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE],
               "iterations_policy": protocol["iterations_policy"], "budget": n})
    leaderboard()
    status(f"Fase 4: `{family}` terminó {len(study.trials)} ensayos Optuna; mejor media Gini {study.best_value:.6f}. "
           f"Estudio SQLite y predicciones conservados, reports/{family}_trials.csv y {family}_study.json. Noviembre sin evaluar.")
