import pandas as pd
import pytest

from bcp_datafest import boost
from bcp_datafest.data import PREDICTORS
from bcp_datafest.io import write_json
from bcp_datafest.variability import within_customer_variability


def test_variability_denominators_and_calendar_gaps():
    df = pd.DataFrame({c: [1, 1, 1, 1] for c in PREDICTORS})
    df["id_cliente"] = [1, 1, 1, 2]
    df["mes"] = [202601, 202602, 202604, 202601]
    df["dias_ultima_interaccion"] = [10, 10, 40, 90]
    table, summary = within_customer_variability(df)
    row = table.set_index("predictor").loc["dias_ultima_interaccion"]
    assert row.changing_customer_fraction == .5
    assert row.changing_fraction_with_history == 1
    assert row.transition_change_fraction == .5
    assert row.monthly_change_fraction == 0
    assert row.median_absolute_delta == 15
    assert summary["gap_counts_months"] == {"1": 1, "2": 1}
    assert table.set_index("predictor").loc["saldo_promedio", "changed_transitions"] == 0


def test_a_reference_is_portable_and_rejects_content_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(boost, "ROOT", tmp_path)
    monkeypatch.setattr(boost, "REPORTS", tmp_path / "reports/boost")
    monkeypatch.setattr(boost, "protocol", lambda: {})
    monkeypatch.setattr(boost, "context", lambda *args: {"data_hashes": {"train.csv": "fixed"}})
    frozen = {"provenance": {"data_hashes": {"train.csv": "fixed"}}, "mean_gini": .26, "components": []}
    write_json(tmp_path / "configs/final.json", frozen)
    write_json(tmp_path / "configs/protocol.json", {"seed": 42})
    write_json(tmp_path / "reports/final_validation.json", {"gini": .23})
    for name in ("selected_development_predictions.csv", "november_predictions.csv"):
        (tmp_path / "reports" / name).write_bytes(b'id_cliente,prediccion\n1,0.2\n')
    boost.candidate_a_reference(tmp_path)
    for path in [*tmp_path.glob("configs/*.json"), *tmp_path.glob("reports/*.json"), *tmp_path.glob("reports/*.csv")]:
        path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    boost.candidate_a_reference(tmp_path)
    write_json(tmp_path / "configs/protocol.json", {"seed": 99})
    with pytest.raises(ValueError, match="reference changed"):
        boost.candidate_a_reference(tmp_path)
