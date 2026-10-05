import copy

import numpy as np
import pandas as pd
import pytest

from bcp_datafest import experiments, identity
from bcp_datafest.io import write_json


def test_json_and_source_hashes_are_portable(tmp_path):
    first, second = tmp_path / "lf.json", tmp_path / "crlf.json"
    first.write_bytes(b'{"b": [2], "a": 1}\n')
    second.write_bytes(b'{\r\n  "a": 1,\r\n  "b": [2]\r\n}\r\n')
    assert identity.json_hash(first) == identity.json_hash(second)
    sources = tmp_path / "src" / "bcp_datafest"
    sources.mkdir(parents=True)
    for name in identity.CODE_FILES:
        (sources / name).write_bytes(b'x = 1\n')
    expected = identity.source_hash(tmp_path)
    for name in identity.CODE_FILES:
        (sources / name).write_bytes(b'x = 1\r\n')
    assert identity.source_hash(tmp_path) == expected
    (sources / "features.py").write_bytes(b'x = 2\n')
    assert identity.source_hash(tmp_path) != expected


def install_fake_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    protocol = {"development_months": [202608], "threads": 1}
    write_json(tmp_path / "configs/boost/protocol.json", protocol)
    source = {"code_sha256": "first", "protocol": protocol, "data_hashes": {"train.csv": "first"}}
    monkeypatch.setattr(experiments, "context", lambda *args: copy.deepcopy(source))
    frame = pd.DataFrame({"id_cliente": [1, 2, 1, 3], "mes": [202607, 202607, 202608, 202608],
                          "objetivo": [0, 1, 0, 1], "x": [0., 1., .2, .8]})
    monkeypatch.setattr(experiments, "load", lambda _: (frame, frame.iloc[-2:], None, None))
    monkeypatch.setattr(experiments, "make_features", lambda df, blocks: df[["x"]])
    fits = []

    class Model:
        def predict_proba(self, x):
            return np.column_stack([1 - x.x, x.x])

    monkeypatch.setattr(experiments, "build", lambda *args: Model())
    monkeypatch.setattr(experiments, "fit_model", lambda *args: fits.append(1))
    return source, fits


def test_cache_retrains_on_code_data_protocol_spec_or_prediction_changes(monkeypatch, tmp_path):
    source, fits = install_fake_experiment(monkeypatch, tmp_path)
    spec = {"name": "same_name", "family": "hgb", "blocks": [], "params": {}}
    first = experiments.evaluate(spec, tmp_path, workspace="boost")
    assert experiments.evaluate(spec, tmp_path, workspace="boost") == first
    assert len(fits) == 1
    for key, value in [("code_sha256", "second"), ("data_hashes", {"train.csv": "second"}),
                       ("protocol", {"new_policy": True})]:
        source[key] = value
        experiments.evaluate(spec, tmp_path, workspace="boost")
    spec = dict(spec, params={"max_iter": 10})
    latest = experiments.evaluate(spec, tmp_path, workspace="boost")
    assert len(fits) == 5
    (tmp_path / latest["prediction_path"]).write_text("damaged", encoding="utf-8")
    with pytest.warns(UserWarning, match="recalculating"):
        experiments.evaluate(spec, tmp_path, workspace="boost")
    assert len(fits) == 6
    with pytest.warns(UserWarning, match="excluded"):
        results, table = experiments.leaderboard(tmp_path, workspace="boost")
    # Same-context configurations are valid, including variants of a run name;
    # only runs from stale code/data/protocol are excluded.
    assert len(results) == 2
    assert not table.empty
    assert (tmp_path / "artifacts/boost/archive").is_dir()


def test_boost_does_not_evaluate_november(monkeypatch, tmp_path):
    install_fake_experiment(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="development"):
        experiments.evaluate({"name": "forbidden"}, tmp_path, [202611], workspace="boost")
