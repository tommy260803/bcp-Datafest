"""Small, causal temporal representations for Candidate B."""
from __future__ import annotations

import numpy as np
import pandas as pd

TEMPORAL_BLOCKS = ("temporal_structure", "temporal_lags", "temporal_v2")
GENERAL = ["observaciones_previas", "sin_historial", "meses_desde_primera_observacion",
           "meses_desde_observacion_previa", "observaciones_previas_3m"]
LAGS = ["interaccion_lag1", "interaccion_lag2", "interaccion_delta1", "interaccion_delta_previa"]
ROLLING = ["interaccion_mean_3m", "interaccion_std_3m", "interaccion_min_3m",
           "interaccion_max_3m", "interaccion_vs_mean_3m"]


def temporal_features(df, block="temporal_v2"):
    """Lags refer to observations; 3m aggregates refer to calendar months [t-3,t)."""
    if block not in TEMPORAL_BLOCKS:
        raise ValueError(f"Unknown temporal block: {block}")
    if df.duplicated(["id_cliente", "mes"]).any():
        raise ValueError("Temporal features require unique customer-month keys")
    if not df.mes.mod(100).between(1, 12).all():
        raise ValueError("Invalid calendar month")
    ordered = df[["id_cliente", "mes", "dias_ultima_interaccion"]].copy()
    ordered["_order"] = np.arange(len(df))
    ordered = ordered.sort_values(["id_cliente", "mes"], kind="stable").reset_index(drop=True)
    month = (ordered.mes // 100) * 12 + ordered.mes % 100
    group = ordered.groupby("id_cliente", sort=False)
    ordinal_group = month.groupby(ordered.id_cliente, sort=False)
    out = pd.DataFrame(index=ordered.index)
    out["observaciones_previas"] = group.cumcount()
    out["sin_historial"] = out.observaciones_previas.eq(0).astype("int8")
    out["meses_desde_primera_observacion"] = month - ordinal_group.transform("min")
    out["meses_desde_observacion_previa"] = month - ordinal_group.shift(1)
    # Unique customer-month keys imply at most three observations in [t-3,t).
    # Shift first, then retain only prior rows in the calendar interval. This
    # avoids per-customer Python loops without assuming a complete sequence.
    recent = pd.DataFrame({
        lag: group.dias_ultima_interaccion.shift(lag).where(
            (month - ordinal_group.shift(lag)).between(1, 3))
        for lag in (1, 2, 3)
    })
    out["observaciones_previas_3m"] = recent.count(axis=1).astype("int8")
    cols = list(GENERAL)
    if block != "temporal_structure":
        out["interaccion_lag1"] = group.dias_ultima_interaccion.shift(1)
        out["interaccion_lag2"] = group.dias_ultima_interaccion.shift(2)
        out["interaccion_delta1"] = ordered.dias_ultima_interaccion - out.interaccion_lag1
        out["interaccion_delta_previa"] = out.interaccion_lag1 - out.interaccion_lag2
        cols += LAGS
    if block == "temporal_v2":
        out["interaccion_mean_3m"] = recent.mean(axis=1)
        out["interaccion_std_3m"] = recent.std(axis=1, ddof=0)
        out["interaccion_min_3m"] = recent.min(axis=1)
        out["interaccion_max_3m"] = recent.max(axis=1)
        out["interaccion_vs_mean_3m"] = ordered.dias_ultima_interaccion - out.interaccion_mean_3m
        cols += ROLLING
    # Positional restoration also works with non-unique DataFrame index labels.
    out["_order"] = ordered._order
    return out.sort_values("_order")[cols].set_axis(df.index)
