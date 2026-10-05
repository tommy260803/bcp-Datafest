from pathlib import Path

import numpy as np
import pandas as pd

from .io import ROOT, digest, write_json, markdown


def validate_submission(path, test, sample=None, *, report_dir=None):
    path = Path(path)
    def require(condition, message):
        if not condition:
            # Preserve the public exception type, but also validate under -O.
            raise AssertionError(message)
    # Validate the serialized file, including the header and numeric parse.
    with path.open("r", encoding="utf-8") as f:
        require(f.readline().strip() == "id_cliente,prediccion", "Invalid submission header")
    df = pd.read_csv(path)
    require(list(df.columns) == ["id_cliente", "prediccion"], "Invalid submission columns")
    require(len(df) == len(test) == 9900, "Submission must contain the 9900 test rows")
    require(df.id_cliente.equals(test.id_cliente), "IDs or original order changed")
    require(not df.id_cliente.duplicated().any(), "Duplicate submission identifiers")
    require(pd.api.types.is_numeric_dtype(df.prediccion), "Predictions must be numeric")
    require(np.isfinite(df.prediccion).all(), "Predictions must be finite")
    require(df.prediccion.between(0, 1).all(), "Predictions must be between zero and one")
    require(df.prediccion.nunique() > 2, "Binary or constant predictions")
    if sample is not None:
        require(not np.array_equal(df.prediccion.to_numpy(), sample.prediccion.to_numpy()), "Example template probabilities were not replaced")
    result = {"passed": True, "file": str(path.resolve()), "rows": len(df), "columns": list(df.columns),
              "identifiers_in_test_order": True, "finite_probabilities": True,
              "min_probability": float(df.prediccion.min()), "max_probability": float(df.prediccion.max()),
              "unique_probabilities": int(df.prediccion.nunique()), "sha256": digest(path)}
    reports = ROOT / "reports" if report_dir is None else Path(report_dir)
    write_json(reports / "submission_validation.json", result)
    markdown(reports / "submission_validation.md", "# Comprobación del CSV exportado\n\n"
             + "\n".join(f"- {k}: `{v}`" for k, v in result.items()) +
             "\n\nCSV UTF-8, coma y punto decimal; precisión de 15 cifras significativas. Probabilidades del modelo elegido. "
             "Hora, vía y representante de entrega pendientes de confirmación del equipo. Gini de diciembre desconocido.\n")
    return result
