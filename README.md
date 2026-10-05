# BCP DataFest

**Archivo de entrega actual:** `outputs/submission.csv` (Candidato A, diciembre,
9.900 filas, verificado en Linux). Para regenerar desde los modelos aislados:

```text
uv run python -m bcp_datafest boost-deliver-a --output outputs/submission.csv
```

Procedencia y comparación con el hash histórico de Windows:
`reports/boost/delivery_a/reproduction.md`. El Gini oficial de diciembre sigue
desconocido.

## Rama experimental boost

La referencia congelada es el **Candidato A**. Los experimentos de **Candidato B**
usan `configs/boost/`, `artifacts/boost/` y `reports/boost/`, con identidad portable
de configuración/código y caché compatible. Instrucciones multiplataforma,
resultados y criterios de decisión: [BOOST.md](BOOST.md).

```text
uv sync --locked
uv run python -m bcp_datafest boost-audit
uv run python -m bcp_datafest boost-experiment --competitive
uv run python -m bcp_datafest boost-diagnose
uv run python -m bcp_datafest boost-hgb
uv run python -m bcp_datafest boost-hgb-stats
uv run python -m bcp_datafest boost-catboost
```

Decisión de la primera ronda: conservar A. La combinación temporal exploratoria
mejora 0.000694 en desarrollo, por debajo del mínimo 0.002; no se evaluó B en
noviembre. Los comandos históricos documentados abajo corresponden al flujo A.

Segunda ronda: ocho configuraciones HGB. El mejor individual alcanza 0.261742,
pero su combinación 75/25 con CatBoost solo mejora 0.000173 sobre A. Los intervalos
exploratorios del individual incluyen cero; se conserva A. Reportes independientes
en `reports/boost/rounds/hgb_round_2/`.

Tercera ronda: seis variantes CatBoost con variables originales. La mejor mejora
solo 0.000070 frente al CatBoost de A y pierde en septiembre/octubre. No se abre
la etapa de combinaciones; se conserva A. Reportes en
`reports/boost/rounds/catboost_round_3/`.

Proceso cliente-mes para primera conversión, con validación temporal y Gini = 2 × AUC − 1. Los predictores actuales se suponen disponibles en el instante de predicción de la competencia; el instante exacto y el evento comercial no están documentados.

Python 3.12 de 64 bits. Instalar uv y ejecutar `uv sync --locked`. En esta máquina se usa uv local: `.\.tools\bin\uv.exe`; Python local: `.python/cpython-3.12.15-windows-x86_64-none/python.exe`.

Los cuatro CSV originales van en `data/`; `--data-dir` permite otra ubicación sin copiar datos. Los originales nunca se modifican. Configuración común en `configs/protocol.json`.

Instalación local equivalente en PowerShell (si no hay Python 3.12/uv en el PATH):

```powershell
python -m pip install --target .tools uv
.\.tools\bin\uv.exe python install 3.12 --install-dir .python --cache-dir .uv-cache
.\.tools\bin\uv.exe sync --locked --python .python/cpython-3.12.15-windows-x86_64-none/python.exe --cache-dir .uv-cache
```

La ruta exacta del Python descargado puede variar si se instala otra revisión 3.12. Los enlaces globales y el registro de Windows no son necesarios para usarlo por ruta. En equipos con uv en PATH: `uv sync --locked` y `uv run python -m bcp_datafest ...`. En este equipo se puede invocar directamente el Python de `.venv` después de sincronizar.

Ejecutar por fases, desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m bcp_datafest audit
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe -m bcp_datafest baseline
.\.venv\Scripts\python.exe -m bcp_datafest experiment
.\.venv\Scripts\python.exe -m bcp_datafest tune --model catboost
.\.venv\Scripts\python.exe -m bcp_datafest tune --model lightgbm
.\.venv\Scripts\python.exe -m bcp_datafest blend
.\.venv\Scripts\python.exe -m bcp_datafest evaluate-final
.\.venv\Scripts\python.exe -m bcp_datafest fit-final
.\.venv\Scripts\python.exe -m bcp_datafest predict --output outputs/submission.csv
.\.venv\Scripts\python.exe -m bcp_datafest validate-submission --file outputs/submission.csv
```

`--data-dir RUTA` es común a todos los comandos. Los estudios de CatBoost y LightGBM pueden ejecutarse en procesos independientes: 4 hilos cada uno, sin compartir SQLite entre máquinas. El presupuesto inicial es 8 ensayos por familia, registrado antes de buscar. `--trials N` amplía un estudio únicamente antes de congelar la selección. No se usan SMOTE, balanceo, calibración ni pesos de clase.

Validación: enero–julio → agosto, enero–agosto → septiembre, enero–septiembre → octubre. Todos los modelos reciben las mismas filas. Se selecciona por media de los tres Gini mensuales; se conserva cada Gini, el peor mes, diagnóstico por historial y AUC global ponderado por composición de diciembre. El historial utiliza exclusivamente meses anteriores; id_cliente y mes se excluyen de los predictores originales. El número de iteraciones es fijo por ensayo; se conserva el del candidato seleccionado.

`blend` prueba pesos 0.25/0.50/0.75 entre las dos mejores familias. Acepta ensamble si mejora la media al menos 0.002 y el límite inferior del intervalo pareado por cliente supera 0. Bootstrap: 500 réplicas agrupando por cliente entre meses. Son diagnósticos exploratorios que no corrigen el optimismo de buscar varios modelos. No se garantiza ganar ni un Gini de diciembre.

`configs/final.json` se congela antes de medir noviembre. `blend` no lo sobrescribe. `evaluate-final` evalúa solo la solución congelada en noviembre, guarda un resultado y lo reutiliza en llamadas posteriores. `fit-final` entrena enero–noviembre con las mismas columnas, parámetros e iteraciones. `predict` carga modelos guardados, construye diciembre, restaura el orden original y exporta 15 cifras significativas. `validate-submission` vuelve a leer el archivo real.

Para regenerar exclusivamente la entrega existente desde modelos conservados, ejecutar `predict` y `validate-submission`. Para reproducir entrenamiento sin repetir búsquedas, conservar `configs/final.json`, `reports/final_validation.json` y datos con hashes idénticos, y ejecutar `fit-final` en una copia sin `artifacts/models/final`. Para repetir todo desde cero, usar un directorio nuevo con código, configuración de protocolo, uv.lock y los mismos datos; no reutilizar la configuración final ni estudios/predicciones de la ejecución previa. Un estudio SQLite guarda ensayos completos/fallidos y un estado del sampler al finalizar; una interrupción abrupta puede dejar ensayos RUNNING y debe documentarse antes de retomarlo.

Los resultados detallados se guardan en `reports/`; cada ensayo conserva configuración, hashes, versiones, tiempo, métricas y predicciones en `artifacts/runs/`. CatBoost se conserva como `.cbm`; LightGBM conserva `.txt`, preprocesamiento y pipeline; scikit-learn usa joblib. El código se identifica también mediante SHA-256 de src/ y protocolo. No cargar joblib de terceros desconocidos.

No se crea un repositorio remoto ni se entrega el archivo al organizador desde este proceso. El equipo debe confirmar representante, hora y vía de entrega del 7 de octubre; el horizonte acordado de desarrollo es 3–6 de octubre. CSV originales, modelos pesados, entornos y credenciales se excluyen de Git. Véase `ESTADO.md` para fases y resultados reales.
