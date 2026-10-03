from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .io import ROOT, digest, write_json, markdown, status

CATS = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
BOOLS = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
NUMS = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
        "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
        "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago", "dias_ultima_interaccion"]
PREDICTORS = NUMS + CATS + BOOLS
FILES = ["train.csv", "test.csv", "sample_submission.csv", "metaData.csv"]


def load(data_dir):
    data_dir = Path(data_dir)
    for name in FILES:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(data_dir / name)
    frames = []
    for name in FILES[:2]:
        df = pd.read_csv(data_dir / name, dtype={c: "string" for c in BOOLS})
        expected = set(PREDICTORS + ["id_cliente", "mes"] + (["objetivo"] if name == "train.csv" else []))
        if set(df.columns) != expected:
            raise ValueError(f"Unexpected schema in {name}: {list(df.columns)}")
        for c in BOOLS:
            if not df[c].isin(["True", "False"]).all():
                raise ValueError(f"Invalid boolean in {name}: {c}")
            df[c] = df[c].map({"True": 1, "False": 0}).astype("int8")
        if df.isna().any().any():
            raise ValueError(f"Missing values in {name}")
        if df.duplicated(["id_cliente", "mes"]).any():
            raise ValueError(f"Duplicate customer-month in {name}")
        for c in NUMS + ["id_cliente", "mes"]:
            if not pd.api.types.is_numeric_dtype(df[c]) or not np.isfinite(df[c]).all():
                raise ValueError(f"Invalid numeric: {name}/{c}")
        if not (df["mes"] % 100).between(1, 12).all():
            raise ValueError("Invalid calendar month")
        frames.append(df)
    train, test = frames
    if not train.objetivo.isin([0, 1]).all():
        raise ValueError("Target must be binary")
    sample = pd.read_csv(data_dir / FILES[2])
    meta = pd.read_csv(data_dir / FILES[3])
    return train, test, sample, meta


def audit(data_dir):
    train, test, sample, meta = load(data_dir)
    assert train.shape == (110100, 25), train.shape
    assert test.shape == (9900, 24), test.shape
    assert sorted(train.mes.unique().tolist()) == list(range(202601, 202612))
    assert test.mes.eq(202612).all()
    assert list(sample.columns) == ["id_cliente", "prediccion"]
    assert sample.id_cliente.equals(test.id_cliente), "Template order mismatch"
    assert not test.id_cliente.duplicated().any()
    assert set(train.columns).issubset(set(meta.column_name))
    positives = train.groupby("id_cliente").objetivo.sum()
    assert positives.le(1).all(), "Multiple positive conversions"
    positive_month = train[train.objetivo.eq(1)].set_index("id_cliente").mes
    all_rows = pd.concat([train.drop(columns="objetivo"), test], ignore_index=True)
    after = all_rows.mes.gt(all_rows.id_cliente.map(positive_month))
    assert not after.any(), "Observation after first conversion"
    variable_clients = all_rows.groupby("id_cliente")[PREDICTORS].nunique().gt(1).sum()
    seen = set()
    monthly = []
    for month, part in train.groupby("mes", sort=True):
        new = ~part.id_cliente.isin(seen)
        monthly.append({"month": int(month), "rows": len(part), "positives": int(part.objetivo.sum()),
                        "positive_rate": float(part.objetivo.mean()), "no_history": int(new.sum()),
                        "no_history_fraction": float(new.mean())})
        seen.update(part.id_cliente)
    new_test = ~test.id_cliente.isin(seen)
    resources = {"logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                 "ram_bytes": psutil.virtual_memory().total, "available_ram_bytes": psutil.virtual_memory().available,
                 "process_rss_bytes": psutil.Process().memory_info().rss}
    result = {"passed": True, "train_shape": list(train.shape), "test_shape": list(test.shape),
              "train_customers": int(train.id_cliente.nunique()), "positive_rows": int(train.objetivo.sum()),
              "positive_rate": float(train.objetivo.mean()), "missing_values": 0, "duplicate_keys": 0,
              "post_conversion_rows": int(after.sum()), "multiple_positive_customers": 0,
              "test_no_history": int(new_test.sum()), "test_no_history_fraction": float(new_test.mean()),
              "monthly": monthly, "varying_clients_per_predictor": {k: int(v) for k, v in variable_clients.items()},
              "hashes": {name: digest(Path(data_dir) / name) for name in FILES}, "resources": resources,
              "types_train": {k: str(v) for k, v in train.dtypes.items()},
              "categories": {c: sorted(all_rows[c].unique().tolist()) for c in CATS}}
    write_json(ROOT / "reports/audit.json", result)
    rows = "\n".join(f"| {r['month']} | {r['rows']} | {r['positives']} | {r['positive_rate']:.6f} | {r['no_history']} |" for r in monthly)
    markdown(ROOT / "reports/audit.md", f"# Auditoría local\n\nTodos los controles de contrato aprobados. Datos originales sin modificaciones.\n\n"
             f"Train {train.shape}; test {test.shape}; {result['train_customers']} clientes en train; "
             f"{result['positive_rows']} positivos ({result['positive_rate']:.6%}). Test sin historial: {int(new_test.sum())}/{len(test)}.\n\n"
             "| Mes | Filas | Positivos | Tasa | Sin historial |\n|---|---:|---:|---:|---:|\n" + rows +
             "\n\nNo hay faltantes, claves duplicadas, múltiples conversiones ni filas posteriores a conversión, incluyendo test. "
             "Variación intracliente verificada en train+test:\n\n" + variable_clients.to_string() +
             "\n\nHashes SHA-256:\n\n" + "\n".join(f"- `{k}`: `{v}`" for k, v in result["hashes"].items()) +
             f"\n\nRecursos: {resources}.\n\nTipos, categorías y conteos completos: `audit.json`.\n")
    status(f"Fases 0–1 cerradas: entorno Python 3.12 y auditoría ejecutable. `audit`: train {train.shape}, test {test.shape}, "
           f"{result['positive_rows']} positivos, {result['train_customers']} clientes, {int(new_test.sum())} test sin historial. "
           "Controles de esquema/orden/primera conversión aprobados; hashes en reports/audit.json. Siguiente: referencias temporales.")
    print(result, flush=True)
    return result

