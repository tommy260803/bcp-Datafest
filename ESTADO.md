# Estado del proyecto

## Boost — ronda 3 CatBoost, 4 de octubre de 2026

- Estado inicial limpio en `boost`; la ronda HGB anterior estaba conservada en Git.
- Presupuesto registrado antes de entrenar en `configs/boost/catboost_round.json`:
  seis variantes CatBoost originales; complejidad/L2 y una variante lenta con
  900 iteraciones. Seed 42, features y bootstrap Bayesian de A intactos. Hasta
  dos ensembles 75/25, únicamente si hay un CatBoost elegible por desarrollo.
- Entrenamientos, manifest y predicciones versionados en
  `artifacts/boost/rounds/catboost_round_3/`; reportes independientes en
  `reports/boost/rounds/catboost_round_3/`. El plan completo participa en la identidad.
- `uv run python -m pytest -q -p no:cacheprovider --basetemp /tmp/opencode/pytest-catboost-round`:
  **25 pruebas aprobadas**, incluyendo presupuesto/features acotados, shortlist
  estable, anchor intacto y rechazo de resultados con procedencias distintas.
- `uv run python -m bcp_datafest boost-catboost`: seis configuraciones completas
  en agosto–octubre, dos referencias de A y HGB shallow fijo. Mejor variante:
  depth3 L2=10, media 0.258420 frente al CatBoost de A 0.258349, delta +0.000070.
  Todas las variantes empeoran septiembre y octubre frente a esa referencia.
- Ninguna configuración pasa la puerta de ganancia y estabilidad; no se abren
  ensembles ni bootstrap para promocionar candidatos descartados. A se reproduce
  con Gini 0.260545 y diferencia máxima de probabilidades ~1e-15.
- **Decisión: conservar A y cerrar esta ronda.** No se amplió el presupuesto,
  no se congeló B, no se evaluó noviembre para B y no se generó otro submission.
- Cierre verificado: nueve cachés reales reutilizados con entrenamiento bloqueado
  y fingerprints idénticos; referencia A íntegra. Sin diferencias en los reportes
  de la ronda HGB anterior ni configuraciones/predicciones de A. `git diff --check`
  sin errores. Cambios de esta ronda sin commit.

## Boost — ronda 2 HGB, 4 de octubre de 2026

- Estado inicial limpio en `boost`; los cambios de la primera ronda ya estaban
  conservados en Git. Configuraciones y resultados históricos de A intactos.
- Presupuesto registrado en `configs/boost/hgb_round.json`: ocho configuraciones
  (original 2, observed_time 3, temporal_lags 3), sin variar learning rate 0.06,
  150 iteraciones, seed 42 ni early stopping. Máximo dos ensembles con pesos de A.
- Ronda y manifest aislados en `artifacts/boost/rounds/hgb_round_2/`; reportes en
  `reports/boost/rounds/hgb_round_2/`. El plan completo participa en la identidad
  y no puede modificarse después de registrar el manifiesto bajo el mismo nombre.
- `uv run python -m pytest -q -p no:cacheprovider --basetemp /tmp/opencode/pytest-hgb-round`:
  22 pruebas aprobadas, incluyendo presupuesto inmutable, selección por estabilidad
  y representación, e independencia de resultados/reportes entre rondas.
- `uv run python -m bcp_datafest boost-hgb`: ocho configuraciones completas en
  agosto–octubre, tres referencias HGB y reproducción de CatBoost A. Mejor HGB:
  observed_time, hojas 7, mínimo 200 muestras, L2 30; Gini 0.261742 frente a su
  referencia 0.257716. Original shallow 0.259734; lags shallow 0.258495.
- La shortlist predefinida seleccionó observed_time shallow y original shallow.
  Ensembles 75/25: 0.260718 (+0.000173 contra A) y 0.260368 (-0.000177). Ninguno
  cumple la ganancia mínima 0.002. No se buscaron más pesos o configuraciones.
- `uv run python -m bcp_datafest boost-hgb-stats`: solo predicciones guardadas;
  HGB mejor contra A, delta 0.001197 e IC95% [-0.004235, 0.006682]; contra su
  referencia, delta 0.004026 e IC95% [-0.002159, 0.009726]. 500 réplicas por
  cliente, todas válidas. Ambos intervalos incluyen cero y no corrigen selección.
- **Decisión: conservar A.** Se retiene HGB observed_time shallow como alternativa
  de desarrollo; no se congeló B, no se evaluó noviembre y no se generó otro CSV.
  La ronda HGB está cerrada; una siguiente ronda requiere una hipótesis diferente.
- Cierre verificado: 22 pruebas aprobadas tras incorporar el diagnóstico; doce
  cachés reales reutilizados con entrenamiento bloqueado y fingerprints idénticos.
  Referencia A íntegra; sin cambios en sus configuraciones/predicciones ni en los
  reportes de la primera ronda. `git diff --check` sin errores. Cambios sin commit.

## Rama boost — 4 de octubre de 2026

- Rama comprobada: `boost`; la modificación preexistente de `.gitignore` se conservó.
- Configuraciones y reportes históricos de A protegidos y sin modificaciones.
  Los artefactos nuevos usan exclusivamente `artifacts/boost/` y `reports/boost/`.
- Identidad experimental v2: hashes JSON lógicos y código normalizado CRLF/LF,
  protocolo completo, cuatro CSV, spec, meses y versiones. Reutilización exige
  además integridad del CSV de predicciones. El leaderboard excluye versiones
  incompatibles explícitamente y conserva un registro; ejecuciones anteriores
  no se mezclan con las actuales.
- `uv run python -m bcp_datafest boost-audit`: enero–octubre, 100.600 filas y
  23.900 clientes; solo cambia `dias_ultima_interaccion`. 78,44% de clientes con
  historial muestran variación y 59,51% de transiciones cambian. No se crean
  rezagos redundantes de saldo, ingresos o productos estáticos.
- Implementados `temporal_structure`, `temporal_lags` y `temporal_v2`, con
  ventanas calendario `[t-3,t)`, NaN para antecedentes ausentes y orden restaurado.
- `uv run python -m pytest -q -p no:cacheprovider --basetemp /tmp/opencode/pytest-boost`:
  **19 pruebas aprobadas**. Para Windows/Linux se documenta el basetemp relativo
  `.pytest-tmp-boost` en BOOST.md.
- `uv run python -m bcp_datafest boost-experiment --competitive`: A reproducido
  Gini 0.260545, diferencia máxima de probabilidades ~1e-15. HGB lags 0.257339;
  HGB v2 0.256387; CatBoost lags 0.255367. CatBoost temporal empeora los tres meses,
  por lo que no se abrió LightGBM ni tuning adicional.
- Una única sustitución del HGB de A por HGB lags, manteniendo pesos 75/25, obtuvo
  0.261239, ganancia 0.000694. No cumple el mínimo 0.002 de aceptación. No se abrió
  búsqueda de pesos; no se congeló ni evaluó B en noviembre.
- Una ejecución competitiva alcanzó el timeout de terminal de 300 segundos tras
  guardar los entrenamientos; al reanudar reutilizó resultados compatibles. Las
  ejecuciones finales completas se realizaron con timeout 600 segundos. Cambios
  posteriores de código/protocolo recalcularon correctamente los experimentos;
  sus versiones antiguas se conservan y quedan fuera del leaderboard vigente.
- `uv run python -m bcp_datafest boost-diagnose`: solo predicciones históricas de
  A; IC95% exploratorio noviembre menos media desarrollo [-0.06248, 0.00715]. La
  composición por historial no explica por sí sola el descenso; no se atribuye
  una causa única ni se ajusta B sobre noviembre.
- **Decisión: mantener Candidato A.** Reportes en `reports/boost/`; explicación y
  reproducción en `BOOST.md`. Próxima ronda solo con una hipótesis nueva y
  presupuesto acotado en agosto–octubre.
- Verificación de cierre: ocho cachés reales reutilizados con entrenamiento
  bloqueado, fingerprints idénticos y referencia A íntegra. `git diff --check`
  sin errores; sin diferencias en configuración final/protocolo/predicciones
  históricas de A. Los cambios quedan en el working tree de boost.

## Registro histórico del Candidato A

Fecha de inicio: 3 de octubre de 2026, America/Lima.

## Fase 0 en curso

- Leído completo `CONTEXTO_BCP_DATAFEST.md` (592 líneas). Directorio inicial: solo contexto y cuatro CSV originales; no existe repositorio Git ni instrucciones AGENTS.md en raíz/ancestros comprobados.
- Python del sistema: 3.13.7, pandas 2.3.2, NumPy 2.2.6, scikit-learn 1.7.2. CatBoost/LightGBM/Optuna ausentes, uv ausente del PATH.
- `python -m pip install --target .tools uv`: terminó correctamente, uv 0.12.23 local.
- `.tools/bin/uv.exe python install 3.12 --install-dir .python --cache-dir .uv-cache`: Python 3.12.15 descargado dentro del proyecto. Las advertencias sobre enlaces y registro global son inocuas para usar su ruta local; no se requiere cambiar el Python del sistema.
- 20 procesadores lógicos; ~297 GB libres en C:. Consulta CIM de RAM/CPU denegada por sandbox; se medirá con psutil. Presupuesto inicial: CPU, 4 hilos, 8 ensayos Optuna por familia, agosto/septiembre/octubre comunes, media mensual de Gini. Noviembre reservado para evaluación final.
- Horizonte del equipo: 3–6 de octubre; entrega indicada 7 de octubre, hora/vía/representante aún pendientes del equipo.

Siguiente paso: sincronizar dependencias en `.venv`, importar bibliotecas y ejecutar auditoría.

## Preparación y verificaciones adicionales

- `uv sync --python .python/cpython-3.12.15-windows-x86_64-none/python.exe --cache-dir .uv-cache` terminó correctamente. Python 3.12.15; versiones fijadas en uv.lock: pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.7.2, CatBoost 1.2.10, LightGBM 4.7.0, Optuna 4.9.0. Importaciones confirmadas.
- Recursos medidos con psutil: 20 hilos lógicos, 14 núcleos, RAM total 16.79 GB y 4.63 GB disponibles al auditar. Los entrenamientos usan 4 hilos por modelo; CatBoost y LightGBM se ejecutan en procesos independientes.
- `python -m pytest -q`: 8 pruebas pasaron; 1 falló al crear el directorio temporal predeterminado por acceso denegado. Recuperación: `python -m pytest -q --basetemp .pytest-tmp`: **9 pasaron**, advertencia de caché pytest denegada. Para posteriores pruebas se deshabilita cacheprovider.
- La aceptación de ensamble se definió antes de consultar resultados competitivos: mejora media >=0.002 y límite inferior del bootstrap pareado por cliente >0. Si no cumple, conservar el mejor individual. 500 réplicas; intervalos exploratorios sin corrección por selección múltiple.
- Referencias scikit-learn reproducidas: Gini HGB agosto 0.259736, septiembre 0.245847, octubre 0.264803. Media 0.256795. Las mínimas diferencias de regresión respecto de la conversación corresponden a implementaciones/versiones, sin forzar coincidencias.
- Durante construcción incremental se añadieron módulos de orquestación. Los hashes de código en las primeras ejecuciones son instantáneas al guardar el resultado; las funciones numéricas models.py/features.py no cambiaron durante esas ejecuciones. El modelo final conservará su hash de código y configuración ya congelados.
- `git init --initial-branch=main` terminó correctamente. Repositorio local inicializado; no se creó remoto, commit ni publicación. Código/configuración/documentación presentes y datos originales ignorados.
- Snapshot de memoria durante entrenamiento CatBoost: proceso principal ~350 MB RSS; disponible ~4 GB. Se mantiene presupuesto CPU acotado.

## Fases 2–4: referencias, variables y entrenamiento competitivo

- `baseline`: logistic media Gini 0.240609; spline 0.239402; HGB 0.256795 en agosto/septiembre/octubre. HGB coincide con la referencia previa en septiembre y octubre (0.245847/0.264803).
- `experiment`: `observed_time` fue el único bloque individual que superó HGB original en media (0.257716 vs 0.256795); agosto 0.258006, septiembre 0.245825, octubre 0.269319. `history` reprodujo el efecto previo (0.250270). `configs/promising_blocks.json` conserva el bloque provisional.
- CatBoost y LightGBM importaron y ejecutaron en CPU. Cada familia terminó 8/8 ensayos Optuna, TPE, seed 42, un proceso secuencial con cuatro hilos; SQLite local y CSV de trials quedaron en `artifacts/studies/` y `reports/`. CatBoost mejor media 0.258349 (`catboost_trial_006`, original, 600 iteraciones, depth 3); LightGBM mejor media 0.257562 (`lightgbm_trial_004`, observed_time, 200 estimadores, 7 hojas). No hubo early stopping, SMOTE, balanceo ni calibración.
- Fase 4 cerrada con predicciones fuera de entrenamiento, tiempos, parámetros, hashes y versiones en cada `artifacts/runs/*/result.json`.

## Fase 5: selección congelada

- `blend` probó pesos 0,25/0,50/0,75 entre las mejores familias. Configuración final `configs/final.json`, SHA-256 `8b797fd55428d35bbed4ec05f56030ea5ca3315eb286fe7456c9a411fa59cc62`.
- Selección: 75% `catboost_trial_006` + 25% `features_observed_time` HGB; media Gini desarrollo 0.260545. Mejora contra el mejor individual 0.002196, bootstrap pareado por cliente 500 réplicas, IC95% [0.000230, 0.004184]. El mejor individual frente al segundo tuvo IC95% que incluye cero; se conserva el ensamble por el criterio predefinido.
- La selección se congeló antes de leer noviembre; `tune` rechaza nuevas búsquedas si existe `configs/final.json`.

## Fase 6: evaluación reservada

- `evaluate-final` entrenó enero–octubre y predijo noviembre una sola vez. Configuración/hash verificados antes de la predicción; no se usó noviembre para iteraciones, variables, pesos o calibración.
- Resultado: 9.500 filas, 1.439 positivos, Gini 0.233079, AUC 0.616540; sin historial Gini 0.191465 (728 filas), con historial 0.233110 (8.772 filas), diagnóstico ponderado por composición de diciembre 0.231954. Reporte en `reports/final_validation.md` y JSON de predicciones en `reports/november_predictions.csv`.

## Fases 7–8: entrega

- `fit-final` reentrenó enero–noviembre (110.100 filas), guardando CatBoost `.cbm`, HGB joblib y metadata/hash en `artifacts/models/final/metadata.json`.
- `predict --output outputs/submission.csv` usó modelos guardados, historial causal train+test y restauración explícita del orden test. `validate-submission` se ejecutó dos veces sobre el CSV serializado.
- Resultado verificado: 9.900 filas; columnas exactas `id_cliente,prediccion`; IDs idénticos y en orden; probabilidades finitas entre 0.0309183115514965 y 0.399970886929378; 9.900 valores únicos; SHA-256 `72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4`. Reportes: `reports/submission_validation.md` y `reports/prediction_provenance.json`.
- El Gini oficial de diciembre sigue desconocido. Pendiente fuera del código: confirmar representante, hora y vía de entrega del 7 de octubre. No se creó remoto ni se envió el archivo al organizador.
- Comprobación de reproducibilidad posterior: `predict --output outputs/submission_repro.csv` y `validate-submission` pasaron; `submission.csv` y `submission_repro.csv` son byte a byte idénticos, ambos SHA-256 `72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4`. `git diff --check` no reporta errores. La segunda ejecución creó únicamente un candidato reproducido ignorado por Git.
- El candidato reproducido se eliminó después de la comparación; `outputs/` contiene únicamente `submission.csv` (246.159 bytes).
- Se intentó preparar un snapshot local con `git add -A`; Windows/sandbox rechazó la creación de `.git/index.lock` (`Permission denied`). No se modificó el índice ni se perdió ningún archivo. El repositorio queda sin commit; el código, reportes y configuración siguen disponibles y el CSV final continúa verificado.
- Cierre técnico confirmado: `uv sync --locked` comprobó las 40 dependencias y `pytest -q -p no:cacheprovider --basetemp .pytest-tmp-final` terminó con 9 pruebas aprobadas. `validate-submission` volvió a aprobar `outputs/submission.csv`: 246.159 bytes, SHA-256 `72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4`.
- Estado operativo: **LISTO PARA ENTREGA**. No se debe volver a ajustar el modelo o consultar etiquetas de diciembre. Falta solamente que el equipo confirme representante, hora y vía de entrega externa.

## 2026-10-03T13:11:50.219467-05:00

Fases 0–1 cerradas: entorno Python 3.12 y auditoría ejecutable. `audit`: train (110100, 25), test (9900, 24), 16567 positivos, 24628 clientes, 1839 test sin historial. Controles de esquema/orden/primera conversión aprobados; hashes en reports/audit.json. Siguiente: referencias temporales.

## 2026-10-03T13:12:07.624891-05:00

Fase 2 cerrada: `baseline` ejecutado en agosto/septiembre/octubre. Resultados en reports/baseline.md y leaderboard.csv; predicciones de validación y métricas por grupos conservadas. Siguiente: comparación controlada de bloques de variables.

## 2026-10-03T13:15:11.729695-05:00

Fase 3: comparaciones controladas terminadas (`experiment`); bloques prometedores provisionales ['observed_time']. Pruebas de temporalidad/orden en tests/. Siguiente: CatBoost/LightGBM y Optuna acotado.

## 2026-10-03T13:17:57.068580-05:00

Fase 4: `lightgbm` terminó 8 ensayos Optuna; mejor media Gini 0.257562. Estudio SQLite y predicciones conservados, reports/lightgbm_trials.csv y lightgbm_study.json. Noviembre sin evaluar.

## 2026-10-03T13:30:51.235580-05:00

Fase 4: `catboost` terminó 8 ensayos Optuna; mejor media Gini 0.258349. Estudio SQLite y predicciones conservados, reports/catboost_trials.csv y catboost_study.json. Noviembre sin evaluar.

## 2026-10-03T13:31:28.889280-05:00

Fase 5 cerrada: configuración congelada en configs/final.json (SHA-256 8b797fd55428d35bbed4ec05f56030ea5ca3315eb286fe7456c9a411fa59cc62). Media Gini desarrollo 0.260545, componentes [{'spec': {'name': 'catboost_trial_006', 'family': 'catboost', 'blocks': [], 'params': {'iterations': 600, 'depth': 3, 'learning_rate': 0.04133944642428183, 'l2_leaf_reg': 31.62562678271973, 'random_strength': 2.3283669032459575, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.2546406282501201}, 'seed': 42}, 'weight': 0.75}, {'spec': {'name': 'features_observed_time', 'family': 'hgb', 'blocks': ['observed_time'], 'params': {}, 'seed': 42}, 'weight': 0.25}]. Bootstrap y combinaciones registrados; siguiente: única evaluación de noviembre con configuración congelada.

## 2026-10-03T13:32:05.984599-05:00

Fase 6 cerrada: noviembre Gini 0.233079, AUC 0.616540, configuración congelada 8b797fd55428d35bbed4ec05f56030ea5ca3315eb286fe7456c9a411fa59cc62. No hubo selección ni early stopping en noviembre. Siguiente: entrenar enero–noviembre.

## 2026-10-03T13:32:39.751088-05:00

Fase 7 entrenamiento cerrado: 110100 filas enero–noviembre; modelos y hashes en artifacts/models/final/metadata.json. Siguiente: predicción de diciembre desde modelos guardados y validación del archivo exportado.

## 2026-10-03T13:32:49.441638-05:00

Fases 7–8 cerradas técnicamente: outputs\submission.csv, 9900 filas, IDs en orden exacto, probabilidades finitas [0,1]. SHA-256 72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4. Pendiente externo: representante, hora y vía de entrega del 7 de octubre. Gini oficial diciembre desconocido.

## 2026-10-03T13:32:50.228405-05:00

Fase 8: validate-submission aprobado. {'passed': True, 'file': 'C:\\Users\\Anthony Garcia\\bcp-datafest\\outputs\\submission.csv', 'rows': 9900, 'columns': ['id_cliente', 'prediccion'], 'identifiers_in_test_order': True, 'finite_probabilities': True, 'min_probability': 0.0309183115514965, 'max_probability': 0.399970886929378, 'unique_probabilities': 9900, 'sha256': '72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4'}

## 2026-10-03T13:33:28.820879-05:00

Fase 8: validate-submission aprobado. {'passed': True, 'file': 'C:\\Users\\Anthony Garcia\\bcp-datafest\\outputs\\submission.csv', 'rows': 9900, 'columns': ['id_cliente', 'prediccion'], 'identifiers_in_test_order': True, 'finite_probabilities': True, 'min_probability': 0.0309183115514965, 'max_probability': 0.399970886929378, 'unique_probabilities': 9900, 'sha256': '72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4'}

## 2026-10-03T13:38:17.650453-05:00

Fases 7–8 cerradas técnicamente: outputs\submission_repro.csv, 9900 filas, IDs en orden exacto, probabilidades finitas [0,1]. SHA-256 72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4. Pendiente externo: representante, hora y vía de entrega del 7 de octubre. Gini oficial diciembre desconocido.

## 2026-10-03T13:38:18.437017-05:00

Fase 8: validate-submission aprobado. {'passed': True, 'file': 'C:\\Users\\Anthony Garcia\\bcp-datafest\\outputs\\submission_repro.csv', 'rows': 9900, 'columns': ['id_cliente', 'prediccion'], 'identifiers_in_test_order': True, 'finite_probabilities': True, 'min_probability': 0.0309183115514965, 'max_probability': 0.399970886929378, 'unique_probabilities': 9900, 'sha256': '72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4'}

## 2026-10-03T13:43:54.655495-05:00

Fase 8: validate-submission aprobado. {'passed': True, 'file': 'C:\\Users\\Anthony Garcia\\bcp-datafest\\outputs\\submission.csv', 'rows': 9900, 'columns': ['id_cliente', 'prediccion'], 'identifiers_in_test_order': True, 'finite_probabilities': True, 'min_probability': 0.0309183115514965, 'max_probability': 0.399970886929378, 'unique_probabilities': 9900, 'sha256': '72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4'}
