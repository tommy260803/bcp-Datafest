from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from .data import load
from .experiments import evaluate
from .features import make_features
from .io import ROOT, digest, read_json, write_json, markdown, provenance, status
from .metrics import metrics
from .models import build, fit_model
from .submission import validate_submission


def configuration(data_dir):
    frozen = read_json(ROOT / "configs/final.json")
    assert frozen["provenance"]["data_hashes"] == provenance(data_dir)["data_hashes"], "Original data hashes changed"
    assert abs(sum(c["weight"] for c in frozen["components"])-1) < 1e-10
    assert all(c["weight"] >= 0 for c in frozen["components"])
    return frozen, digest(ROOT / "configs/final.json")


def evaluate_final(data_dir):
    frozen, sha = configuration(data_dir)
    report = ROOT / "reports/final_validation.json"
    if report.exists():
        previous = read_json(report)
        assert previous["frozen_config_sha256"] == sha
        print("November already evaluated. Existing result:", previous, flush=True)
        return previous
    month = read_json(ROOT / "configs/protocol.json")["final_validation_month"]
    probs = None
    reference = None
    for component in frozen["components"]:
        spec = dict(component["spec"])
        spec["name"] = "final_validation_" + spec["name"]
        evaluate(spec, data_dir, [month])
        part = pd.read_csv(ROOT / "artifacts/runs" / spec["name"] / "validation.csv")
        if reference is not None:
            assert reference[["id_cliente", "mes", "objetivo"]].equals(part[["id_cliente", "mes", "objetivo"]])
        else:
            reference = part
            probs = np.zeros(len(part))
        probs += component["weight"] * part.prediccion.to_numpy()
    result = {"frozen_config_sha256": sha, "frozen_at_lima": frozen["frozen_at_lima"],
              "metrics": metrics(reference.objetivo, probs, reference.sin_historial, frozen["test_no_history_fraction"]),
              "training_months": list(range(202601, 202611)), "evaluation_month": month,
              "provenance": provenance(data_dir), "development_mean_gini": frozen["mean_gini"]}
    write_json(report, result)
    reference["prediccion"] = probs
    reference.to_csv(ROOT / "reports/november_predictions.csv", index=False, float_format="%.15g")
    markdown(ROOT / "reports/final_validation.md", "# Evaluación final noviembre\n\n"
             f"Configuración congelada {sha} desde {frozen['frozen_at_lima']}. Train enero–octubre; evaluación noviembre. "
             "Variables, pesos, semillas e iteraciones fijados en desarrollo.\n\n"
             f"Gini global {result['metrics']['gini']:.6f}; AUC {result['metrics']['auc']:.6f}; "
             f"media desarrollo {frozen['mean_gini']:.6f}.\n\n"
             + "\n".join(f"- {k}: {v}" for k, v in result["metrics"].items()) +
             "\n\nUna evaluación de un solo mes; no es garantía del Gini de diciembre. Noviembre tiene menor proporción "
             "sin historial que diciembre; el AUC global reponderado es solo diagnóstico de composición. "
             "No se ajustan hiperparámetros tras consultar este resultado.\n")
    status(f"Fase 6 cerrada: noviembre Gini {result['metrics']['gini']:.6f}, AUC {result['metrics']['auc']:.6f}, "
           f"configuración congelada {sha}. No hubo selección ni early stopping en noviembre. Siguiente: entrenar enero–noviembre.")
    print(result, flush=True)
    return result


def fit_final(data_dir):
    frozen, sha = configuration(data_dir)
    validation = read_json(ROOT / "reports/final_validation.json")
    assert validation["frozen_config_sha256"] == sha
    train, _, _, _ = load(data_dir)
    directory = ROOT / "artifacts/models/final"
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        existing = read_json(meta_path)
        assert existing["frozen_config_sha256"] == sha
        assert existing["provenance"]["code_sha256"] == provenance(data_dir)["code_sha256"], "Code changed since final fit"
        print("Final models already exist; preserving them.", flush=True)
        return existing
    entries = []
    for i, component in enumerate(frozen["components"]):
        spec = component["spec"]
        x = make_features(train, spec["blocks"])
        assert list(x.columns) == frozen["feature_columns"][spec["name"]]
        model = build(spec, x)
        start = time.perf_counter()
        with threadpool_limits(limits=4):
            fit_model(model, spec, x, train.objetivo)
        prefix = directory / f"component_{i}"
        if spec["family"] == "catboost":
            path = prefix.with_suffix(".cbm")
            model.save_model(str(path))
        else:
            path = prefix.with_suffix(".joblib")
            joblib.dump(model, path)
            if spec["family"] == "lightgbm":
                native = prefix.with_suffix(".txt")
                model.named_steps["model"].booster_.save_model(str(native))
                joblib.dump(model.named_steps["prep"], prefix.with_suffix(".preprocessor.joblib"))
        entries.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "spec": spec,
                        "weight": component["weight"], "columns": list(x.columns), "seconds": time.perf_counter()-start})
        print(f"Final fit {spec['name']}: {entries[-1]['seconds']:.1f}s; saved {path}", flush=True)
    metadata = {"frozen_config_sha256": sha, "components": entries, "train_rows": len(train),
                "train_months": sorted(train.mes.unique().tolist()), "provenance": provenance(data_dir),
                "november_gini": validation["metrics"]["gini"]}
    write_json(meta_path, metadata)
    status(f"Fase 7 entrenamiento cerrado: {len(train)} filas enero–noviembre; modelos y hashes en artifacts/models/final/metadata.json. "
           "Siguiente: predicción de diciembre desde modelos guardados y validación del archivo exportado.")
    return metadata


def predict(data_dir, output):
    frozen, sha = configuration(data_dir)
    train, test, sample, _ = load(data_dir)
    directory = ROOT / "artifacts/models/final"
    metadata = read_json(directory / "metadata.json")
    assert metadata["frozen_config_sha256"] == sha
    assert metadata["provenance"]["code_sha256"] == provenance(data_dir)["code_sha256"]
    combined = pd.concat([train.drop(columns="objetivo"), test], ignore_index=True)
    test_positions = np.arange(len(train), len(combined))
    p = np.zeros(len(test))
    for component in metadata["components"]:
        spec = component["spec"]
        x = make_features(combined, spec["blocks"]).iloc[test_positions]
        assert list(x.columns) == component["columns"]
        path = ROOT / component["path"]
        assert digest(path) == component["sha256"]
        if spec["family"] == "catboost":
            model = CatBoostClassifier()
            model.load_model(str(path))
        else:
            model = joblib.load(path)
        with threadpool_limits(limits=4):
            p += component["weight"] * model.predict_proba(x)[:, 1]
    # The test input positions were retained explicitly through history sorting.
    assert combined.iloc[test_positions].id_cliente.reset_index(drop=True).equals(test.id_cliente)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_cliente": test.id_cliente.to_numpy(), "prediccion": p}).to_csv(output, index=False, encoding="utf-8", float_format="%.15g")
    result = validate_submission(output, test, sample)
    write_json(ROOT / "reports/prediction_provenance.json", {"validation": result, "frozen_config_sha256": sha,
               "final_metadata_sha256": digest(directory / "metadata.json"), "provenance": provenance(data_dir)})
    status(f"Fases 7–8 cerradas técnicamente: {output}, {result['rows']} filas, IDs en orden exacto, probabilidades finitas [0,1]. "
           f"SHA-256 {result['sha256']}. Pendiente externo: representante, hora y vía de entrega del 7 de octubre. Gini oficial diciembre desconocido.")
    print(result, flush=True)
    return result
