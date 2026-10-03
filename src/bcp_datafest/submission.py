from pathlib import Path

import numpy as np
import pandas as pd

from .io import ROOT, digest, write_json, markdown


def validate_submission(path, test, sample=None):
    path = Path(path)
    # Validate the serialized file, including the header and numeric parse.
    with path.open("r", encoding="utf-8") as f:
        assert f.readline().strip() == "id_cliente,prediccion"
    df = pd.read_csv(path)
    assert list(df.columns) == ["id_cliente", "prediccion"]
    assert len(df) == len(test) == 9900
    assert df.id_cliente.equals(test.id_cliente), "IDs or original order changed"
    assert not df.id_cliente.duplicated().any()
    assert pd.api.types.is_numeric_dtype(df.prediccion)
    assert np.isfinite(df.prediccion).all()
    assert df.prediccion.between(0, 1).all()
    assert df.prediccion.nunique() > 2, "Binary or constant predictions"
    if sample is not None:
        assert not np.array_equal(df.prediccion.to_numpy(), sample.prediccion.to_numpy())
    result = {"passed": True, "file": str(path.resolve()), "rows": len(df), "columns": list(df.columns),
              "identifiers_in_test_order": True, "finite_probabilities": True,
              "min_probability": float(df.prediccion.min()), "max_probability": float(df.prediccion.max()),
              "unique_probabilities": int(df.prediccion.nunique()), "sha256": digest(path)}
    write_json(ROOT / "reports/submission_validation.json", result)
    markdown(ROOT / "reports/submission_validation.md", "# Comprobación del CSV exportado\n\n"
             + "\n".join(f"- {k}: `{v}`" for k, v in result.items()) +
             "\n\nCSV UTF-8, coma y punto decimal; precisión de 15 cifras significativas. Probabilidades del modelo elegido. "
             "Hora, vía y representante de entrega pendientes de confirmación del equipo. Gini de diciembre desconocido.\n")
    return result
