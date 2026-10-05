"""Development-only audit: no future values decide the temporal feature set."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NUMS, PREDICTORS, load
from .identity import context
from .io import ROOT, markdown, read_json, write_json


def within_customer_variability(df):
    ordered = df.sort_values(["id_cliente", "mes"], kind="stable").reset_index(drop=True)
    if ordered.duplicated(["id_cliente", "mes"]).any():
        raise ValueError("Variability audit requires unique customer-month keys")
    group = ordered.groupby("id_cliente", sort=False)
    sizes = group.size()
    eligible = sizes.ge(2)
    months = (ordered.mes // 100) * 12 + ordered.mes % 100
    gaps = months - months.groupby(ordered.id_cliente).shift(1)
    transitions, monthly = gaps.notna(), gaps.eq(1)
    rows = []
    for col in PREDICTORS:
        distinct = group[col].nunique()
        changed = ordered[col].ne(group[col].shift(1)) & transitions
        row = {"predictor": col, "customers": len(sizes), "customers_with_history": int(eligible.sum()),
               "changing_customers": int(distinct.gt(1).sum()),
               "changing_customer_fraction": float(distinct.gt(1).mean()),
               "changing_fraction_with_history": float(distinct[eligible].gt(1).mean()) if eligible.any() else None,
               "mean_distinct_values": float(distinct.mean()),
               "mean_distinct_with_history": float(distinct[eligible].mean()) if eligible.any() else None,
               "transitions": int(transitions.sum()), "changed_transitions": int(changed.sum()),
               "transition_change_fraction": float(changed.sum() / transitions.sum()) if transitions.any() else None,
               "monthly_transitions": int(monthly.sum()),
               "monthly_change_fraction": float(changed[monthly].mean()) if monthly.any() else None}
        if col in NUMS:
            delta = (ordered[col] - group[col].shift(1))[transitions].astype(float)
            row.update(mean_signed_delta=float(delta.mean()) if len(delta) else None,
                       median_absolute_delta=float(delta.abs().median()) if len(delta) else None,
                       p90_absolute_delta=float(delta.abs().quantile(.9)) if len(delta) else None,
                       median_absolute_delta_when_changed=float(delta[delta.ne(0)].abs().median()) if delta.ne(0).any() else None)
        rows.append(row)
    return pd.DataFrame(rows), {"rows": len(df), "customers": len(sizes),
                               "gap_counts_months": {str(int(k)): int(v) for k, v in gaps.dropna().value_counts().sort_index().items()}}


def audit_variability(data_dir):
    protocol = read_json(ROOT / "configs/boost/protocol.json")
    train, _, _, _ = load(data_dir)
    cutoff = max(protocol["development_months"])
    table, summary = within_customer_variability(train[train.mes.le(cutoff)])
    measured = table.loc[table.changing_customers.gt(0), "predictor"].tolist()
    result = {"cutoff_month": cutoff, "summary": summary, "dynamic_predictors": measured,
              "configured_dynamic_predictors": protocol["dynamic_predictors"],
              "context": context(data_dir, protocol), "predictors": table.replace({np.nan: None}).to_dict("records")}
    reports = ROOT / "reports/boost"
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "variability.csv", index=False)
    write_json(reports / "variability.json", result)
    markdown(reports / "variability.md", f"# Variabilidad intracliente — desarrollo hasta {cutoff}\n\n"
             "No se utilizan valores de noviembre/diciembre para seleccionar predictores dinámicos. "
             "Las frecuencias mensuales consideran únicamente pares consecutivos en calendario; "
             "los cambios absolutos usan todas las transiciones observadas.\n\n"
             f"Resumen: {summary}.\n\nPredictores que cambian: {measured}.\n\n```text\n"
             + table.to_string(index=False) + "\n```\n")
    print(f"Variability through {cutoff}: dynamic predictors={measured}; {summary}", flush=True)
    return result
