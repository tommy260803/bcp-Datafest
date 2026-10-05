from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PREDICTORS
from .temporal import TEMPORAL_BLOCKS, temporal_features

BLOCKS = ["economic", "linkage", "activity", "history", "observed_time", *TEMPORAL_BLOCKS]


def history(df):
    """Strictly past observations, stable restoration of arbitrary input order."""
    if df.duplicated(["id_cliente", "mes"]).any():
        raise ValueError("History requires unique customer-month keys")
    ordered = df[["id_cliente", "mes", "dias_ultima_interaccion"]].copy()
    ordered["_order"] = np.arange(len(df))
    ordered = ordered.sort_values(["id_cliente", "mes"], kind="stable")
    g = ordered.groupby("id_cliente", sort=False)
    ordered["observaciones_previas"] = g.cumcount()
    ordered["sin_historial"] = ordered.observaciones_previas.eq(0).astype("int8")
    ordered["interaccion_previa"] = g.dias_ultima_interaccion.shift(1).fillna(ordered.dias_ultima_interaccion)
    ordered["media_interaccion_previa"] = g.dias_ultima_interaccion.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(ordered.dias_ultima_interaccion)
    ordered["cambio_interaccion"] = ordered.dias_ultima_interaccion - ordered.interaccion_previa
    ordinal = (ordered.mes // 100) * 12 + ordered.mes % 100
    first = ordinal.groupby(ordered.id_cliente).transform("min")
    ordered["meses_desde_primera_observacion"] = ordinal - first
    # No aggregate of future values or target is used. Minimum month is causal
    # because rows are sorted and all later rows have a greater month.
    cols = ["observaciones_previas", "sin_historial", "interaccion_previa", "media_interaccion_previa",
            "cambio_interaccion", "meses_desde_primera_observacion"]
    return ordered.sort_values("_order")[cols].set_axis(df.index)


def make_features(df, blocks=()):
    unknown = set(blocks) - set(BLOCKS)
    if unknown:
        raise ValueError(f"Unknown blocks: {unknown}")
    temporal = set(blocks) & set(TEMPORAL_BLOCKS)
    if len(temporal) > 1 or (temporal and set(blocks) & {"history", "observed_time"}):
        raise ValueError("Temporal blocks are alternative representations; do not duplicate history columns")
    x = df[PREDICTORS].copy()
    if "economic" in blocks:
        monthly_income = df.ingresos / 12
        x["ingreso_mensual_estimado"] = monthly_income
        x["saldo_relativo_ingreso_mensual"] = df.saldo_promedio / monthly_income.where(monthly_income.abs() > 1e-8)
    if "linkage" in blocks:
        x["productos_flags"] = df.tiene_tarjeta_credito + df.tiene_prestamo + df.tiene_seguro
        x["combinacion_productos"] = df.tiene_tarjeta_credito + 2 * df.tiene_prestamo + 4 * df.tiene_seguro
        x["productos_por_flags"] = df.numero_productos * x.productos_flags
        x["riesgo_productos"] = df.banda_riesgo.astype(str) + "_" + df.numero_productos.astype(str)
    if "activity" in blocks:
        x["recencia_por_movil"] = df.dias_ultima_transaccion * df.activo_movil
        x["productos_por_movil"] = df.numero_productos * df.activo_movil
        x["visitas_por_movil"] = df.visitas_web_ultimos_90_dias * df.activo_movil
    if "history" in blocks or "observed_time" in blocks:
        h = history(df)
        if "history" in blocks:
            x = pd.concat([x, h.drop(columns="meses_desde_primera_observacion")], axis=1)
        if "observed_time" in blocks:
            x["meses_desde_primera_observacion"] = h.meses_desde_primera_observacion
    if temporal:
        x = pd.concat([x, temporal_features(df, next(iter(temporal)))], axis=1)
    return x
