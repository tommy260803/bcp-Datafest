import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from bcp_datafest.data import load
from bcp_datafest.features import make_features
from bcp_datafest.io import ROOT
from bcp_datafest.models import build, fit_model
from bcp_datafest.submission import validate_submission


@pytest.mark.parametrize("family", ["logistic", "hgb", "catboost", "lightgbm"])
def test_training_only_categories_and_unseen_inference(family):
    train, _, _, _ = load(ROOT / "data")
    small = train.iloc[:300]
    x = make_features(small, ["linkage"])
    params = {"catboost": {"iterations": 5, "depth": 3}, "lightgbm": {"n_estimators": 5, "min_child_samples": 10},
              "hgb": {"max_iter": 5, "min_samples_leaf": 10}, "logistic": {}}[family]
    spec = {"family": family, "params": params, "seed": 42}
    with threadpool_limits(limits=2):
        model = fit_model(build(spec, x, threads=2), spec, x, small.objetivo)
        unseen = x.iloc[:2].copy()
        unseen["ocupacion"] = "category_never_in_training"
        p = model.predict_proba(unseen)[:, 1]
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    if family != "catboost":
        cat_encoder = model.named_steps["prep"].named_transformers_["cat"]
        assert "category_never_in_training" not in cat_encoder.categories_[0]


def test_serialized_submission_rejects_reordered_ids(tmp_path):
    _, test, sample, _ = load(ROOT / "data")
    output = pd.DataFrame({"id_cliente": test.id_cliente, "prediccion": np.linspace(.01, .99, len(test))})
    path = tmp_path / "invalid.csv"
    output.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(AssertionError, match="IDs or original order"):
        validate_submission(path, test, sample)
    output.loc[0, "prediccion"] = np.inf
    output.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        validate_submission(path, test, sample)
