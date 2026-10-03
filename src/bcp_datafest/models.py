from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, SplineTransformer


def categories(x):
    return x.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def build(spec, x, threads=4):
    family = spec["family"]
    params = spec.get("params", {})
    seed = spec.get("seed", 42)
    cats = categories(x)
    numeric = [c for c in x.columns if c not in cats]
    if family == "catboost":
        return CatBoostClassifier(cat_features=cats, loss_function="Logloss", thread_count=threads,
                                  random_seed=seed, verbose=False, allow_writing_files=False,
                                  **params)
    if family in ["logistic", "spline"]:
        num_steps = [("impute", SimpleImputer(strategy="median"))]
        if family == "spline":
            # Preserve binary flags as direct standardized predictors.
            binary = [c for c in numeric if x[c].dropna().isin([0, 1]).all()]
            curves = [c for c in numeric if c not in binary]
            prep = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                ("curves", Pipeline(num_steps + [("spline", SplineTransformer(n_knots=4, degree=2, knots="quantile", extrapolation="linear")),
                                                 ("scale", StandardScaler())]), curves),
                ("binary", StandardScaler(), binary)])
        else:
            prep = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                ("num", Pipeline(num_steps + [("scale", StandardScaler())]), numeric)])
        return Pipeline([("prep", prep), ("model", LogisticRegression(C=0.1, max_iter=400, solver="lbfgs", **params))])
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
        ("numeric", "passthrough", numeric)], verbose_feature_names_out=False)
    mask = [True] * len(cats) + [False] * len(numeric)
    if family == "hgb":
        defaults = dict(max_iter=150, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=10,
                        learning_rate=0.06, early_stopping=False, random_state=seed, categorical_features=mask)
        defaults.update(params)
        model = HistGradientBoostingClassifier(**defaults)
    elif family == "lightgbm":
        model = LGBMClassifier(objective="binary", random_state=seed, n_jobs=threads, verbosity=-1, **params)
    else:
        raise ValueError(f"Unknown family: {family}")
    return Pipeline([("prep", prep), ("model", model)])


def fit_model(model, spec, x, y):
    if spec["family"] == "lightgbm":
        model.fit(x, y, model__categorical_feature=list(range(len(categories(x)))))
    else:
        model.fit(x, y)
    return model
