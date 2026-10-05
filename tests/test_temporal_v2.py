import numpy as np
import pandas as pd
import pytest

from bcp_datafest.temporal import temporal_features
from bcp_datafest.features import make_features
from bcp_datafest.metrics import gini, metrics
from bcp_datafest.ensemble import paired_bootstrap


def observations():
    return pd.DataFrame({"id_cliente": [1, 1, 2, 1, 1],
                         "mes": [202611, 202612, 202612, 202702, 202703],
                         "dias_ultima_interaccion": [10, 20, 800, 40, 60],
                         "objetivo": [0, 0, 1, 0, 1]}, index=[8, 4, 3, 9, 2])


def test_calendar_window_lags_and_missing_history():
    out = temporal_features(observations())
    assert out.loc[8, "sin_historial"] == 1
    assert np.isnan(out.loc[8, "interaccion_lag1"])
    assert out.loc[3, "observaciones_previas_3m"] == 0
    assert out.loc[9, "meses_desde_observacion_previa"] == 2
    assert out.loc[9, "interaccion_lag2"] == 10
    assert out.loc[9, "interaccion_mean_3m"] == 15  # Nov, Dec in [Nov, Feb)
    assert out.loc[9, "interaccion_std_3m"] == 5
    assert out.loc[2, "interaccion_mean_3m"] == 30  # Dec, Feb; Nov excluded
    assert out.loc[2, "interaccion_delta_previa"] == 20


def test_future_labels_other_customers_and_order_do_not_contaminate():
    df = observations()
    old = temporal_features(df[df.mes.le(202612)])
    changed = df.copy()
    changed.loc[changed.mes.gt(202612), "dias_ultima_interaccion"] = -10000
    changed["objetivo"] = 1 - changed.objetivo
    pd.testing.assert_frame_equal(old, temporal_features(changed).loc[old.index])
    added = pd.concat([df, pd.DataFrame({"id_cliente": [1], "mes": [202704],
                                       "dias_ultima_interaccion": [9999], "objetivo": [1]}, index=[100])])
    pd.testing.assert_frame_equal(temporal_features(df), temporal_features(added).loc[df.index])
    changed = df.copy()
    changed.loc[changed.id_cliente.eq(2), "dias_ultima_interaccion"] = 9999
    baseline = temporal_features(df)
    ids = df[df.id_cliente.eq(1)].index
    pd.testing.assert_frame_equal(baseline.loc[ids], temporal_features(changed).loc[ids])
    pd.testing.assert_frame_equal(baseline, temporal_features(df.sample(frac=1, random_state=42)).loc[df.index])
    duplicate_index = df.set_axis([0] * len(df))
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True), temporal_features(duplicate_index).reset_index(drop=True))


def test_current_values_only_affect_current_differences_not_history():
    df = observations()
    old = temporal_features(df)
    df.loc[9, "dias_ultima_interaccion"] = 100
    new = temporal_features(df)
    for col in ["interaccion_lag1", "interaccion_lag2", "interaccion_mean_3m", "interaccion_std_3m"]:
        assert new.loc[9, col] == old.loc[9, col]
    assert new.loc[9, "interaccion_vs_mean_3m"] - old.loc[9, "interaccion_vs_mean_3m"] == 60


def test_duplicate_keys_and_duplicate_feature_representations_rejected():
    df = observations()
    with pytest.raises(ValueError, match="unique"):
        temporal_features(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError, match="alternative"):
        make_features(df, ["history", "temporal_v2"])


def test_one_class_and_zero_weight_classes_are_undefined():
    assert gini([0, 1], [.2, .8], [1, 0]) is None
    result = metrics([0, 0], [.1, .2], [True, False], .2)
    assert result["gini"] is None and result["auc"] is None and result["top20_recall"] is None
    df = pd.DataFrame({"id_cliente": [1, 2], "mes": [202608] * 2, "objetivo": [0, 1]})
    result = paired_bootstrap(df, np.array([.1, .9]), np.array([.2, .8]), replicates=100)
    assert result["invalid_replicates"] > 0
    assert result["valid_replicates"] + result["invalid_replicates"] == 100
    assert result["ci95"] == [0., 0.]
