import numpy as np
import pandas as pd
import pytest

from bcp_datafest.features import history


def fixture():
    return pd.DataFrame({"id_cliente": [1, 2, 1, 1, 2], "mes": [202611, 202612, 202612, 202701, 202611],
                         "dias_ultima_interaccion": [10, 90, 20, 999, 80]})


def test_future_changes_do_not_change_past():
    df = fixture()
    old = history(df[df.mes <= 202612])
    changed = df.copy()
    changed.loc[changed.mes == 202701, "dias_ultima_interaccion"] = -1000
    pd.testing.assert_frame_equal(old, history(changed).loc[old.index])
    extra = pd.concat([df, pd.DataFrame({"id_cliente": [1], "mes": [202702], "dias_ultima_interaccion": [8000]})], ignore_index=True)
    pd.testing.assert_frame_equal(old, history(extra).loc[old.index])


def test_first_row_no_history_and_calendar_gap():
    h = history(fixture())
    assert h.loc[0, "sin_historial"] == h.loc[4, "sin_historial"] == 1
    assert h.loc[2, "interaccion_previa"] == 10
    assert h.loc[3, "media_interaccion_previa"] == 15
    assert h.loc[3, "meses_desde_primera_observacion"] == 2
    df = pd.DataFrame({"id_cliente": [1, 1], "mes": [202612, 202702], "dias_ultima_interaccion": [2, 4]})
    h = history(df)
    assert h.loc[1, "observaciones_previas"] == 1
    assert h.loc[1, "meses_desde_primera_observacion"] == 2


def test_history_preserves_input_order():
    df = fixture()
    expected = history(df)
    shuffled = df.sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(expected, history(shuffled).loc[df.index])


def test_duplicate_customer_month_rejected():
    df = fixture()
    with pytest.raises(ValueError):
        history(pd.concat([df, df.iloc[:1]]))

