import copy

import numpy as np
import pandas as pd
import pytest

from bcp_datafest import deliver_a
from bcp_datafest.identity import logical_hash
from bcp_datafest.io import digest


def test_export_is_portable_validates_actual_file_and_preserves_existing_output(tmp_path):
    test = pd.DataFrame({"id_cliente": np.arange(9900, dtype=np.int64)})
    sample = pd.DataFrame({"id_cliente": test.id_cliente, "prediccion": .15})
    p = np.linspace(.01, .99, len(test))
    output = tmp_path / "submission.csv"
    reports = tmp_path / "isolated_reports"
    first = deliver_a.export_submission(output, test, p, sample, reports)
    assert first["passed"] and first["rows"] == 9900
    assert b'\r\n' not in output.read_bytes()
    assert not (tmp_path / "reports/submission_validation.json").exists()
    assert deliver_a.export_submission(output, test, p, sample, reports)["sha256"] == first["sha256"]
    with pytest.raises(FileExistsError, match="differs"):
        deliver_a.export_submission(output, test, p + .001, sample, reports)
    assert digest(output) == first["sha256"]
    assert not list(tmp_path.glob("*.tmp"))


def test_model_bundle_identity_and_integrity_are_required(monkeypatch, tmp_path):
    monkeypatch.setattr(deliver_a, "ROOT", tmp_path)
    model = tmp_path / "own_model.joblib"
    model.write_bytes(b'owned model bytes')
    spec = {"name": "hgb", "family": "hgb", "blocks": [], "params": {}}
    frozen = {"components": [{"spec": spec, "weight": 1.}], "feature_columns": {"hgb": ["x"]}}
    recipe = {"candidate": "A", "context": {"code": "current"}}
    metadata = {"recipe": recipe, "fingerprint": logical_hash(recipe),
                "components": [{"spec": spec, "weight": 1., "columns": ["x"], "path": model.name,
                                "sha256": digest(model)}]}
    deliver_a.verify_models(metadata, recipe, frozen)
    with pytest.raises(ValueError, match="recipe"):
        deliver_a.verify_models(metadata, {"candidate": "A", "context": {"code": "changed"}}, frozen)
    changed = copy.deepcopy(metadata)
    changed["components"][0]["columns"] = ["wrong"]
    with pytest.raises(ValueError, match="columns"):
        deliver_a.verify_models(changed, recipe, frozen)
    model.write_bytes(b'damaged model')
    with pytest.raises(ValueError, match="damaged"):
        deliver_a.verify_models(metadata, recipe, frozen)
