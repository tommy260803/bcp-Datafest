# Rama boost — Candidato B

## Decisión actual

**Conservar Candidato A.** Las rondas de features temporales y ajuste HGB no justifican
sustituir la solución congelada. No se evaluó ningún modelo B en noviembre, ni se
generó una entrega nueva. La configuración y los reportes históricos de A se
conservan sin modificaciones.

En la primera ronda, la combinación adicional, 75% CatBoost original y 25% HGB con rezagos,
mejora la media de 0.260545 a 0.261239. Su ganancia de **0.000694** no alcanza el
mínimo 0.002 registrado para aceptar un ensemble; por eso no se promociona ni se
abre una búsqueda de pesos. No se calculó un intervalo para esta combinación:
falló primero el umbral de ganancia. Los intervalos existentes son exploratorios,
sin corrección por selección de modelos.

## Reproducción en Linux y Windows

Desde la raíz, con Python 3.12 y uv:

```text
uv sync --locked
uv run python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-boost
uv run python -m bcp_datafest boost-audit
uv run python -m bcp_datafest boost-experiment --competitive
uv run python -m bcp_datafest boost-diagnose
uv run python -m bcp_datafest boost-hgb
uv run python -m bcp_datafest boost-hgb-stats
```

Todos los comandos aceptan `--data-dir RUTA`. No requieren bash ni PowerShell
específicos. `boost-experiment` sin `--competitive` ejecuta solo la ronda HGB y la
reproducción de A; con el flag añade una comparación CatBoost de presupuesto fijo.
LightGBM se abre únicamente si esa comparación CatBoost supera el criterio de
ganancia/estabilidad. En esta ejecución no se abrió. En la primera ronda no se hizo
tuning nuevo; la segunda registra exactamente ocho configuraciones HGB. No se
incorporó XGBoost.

Los nuevos archivos están aislados en:

- `configs/boost/protocol.json`: cortes, presupuesto y criterios de decisión.
- `artifacts/boost/runs/<nombre>/<fingerprint>/`: configuración, resultado y
  predicciones. Ignorado por Git.
- `reports/boost/`: auditoría, leaderboard, comparaciones y diagnóstico.
- `configs/boost/hgb_round.json`: presupuesto y configuraciones de la ronda 2.
- `artifacts/boost/rounds/hgb_round_2/`: manifiesto registrado antes de entrenar,
  runs versionados, ensembles y comparaciones. Ignorado por Git.
- `reports/boost/rounds/hgb_round_2/`: reportes independientes de la ronda 2.

Cambiar ramas no restaura archivos ignorados. Para operar el flujo histórico de A
se debe usar su versión de código y sus modelos conservados; no mezclar sus
metadatos antiguos con código experimental. La reproducción de A en boost usa
los mismos parámetros y features, y guarda resultados separados.

## Identidad y caché

Los experimentos nuevos usan `experiment-v2`:

- JSON canónico: claves ordenadas, separadores fijos, UTF-8 y sin NaN.
- Código normalizado: saltos LF/CRLF equivalentes y nombres de módulos estables.
- Hashes físicos de los cuatro CSV originales.
- Protocolo completo, spec, meses y versiones de bibliotecas relevantes.
- Python mayor/menor en la identidad; revisión exacta y plataforma en `runtime`.

El manifiesto identifica los módulos de lectura, features, modelos, métricas,
evaluación, selección y funciones auxiliares. Si se añade una dependencia numérica
nueva, debe incluirse en `identity.CODE_FILES`/`PACKAGES`.

Reutilizar una ejecución exige identidad idéntica y hash válido del CSV de
predicciones. Los cambios crean otra ejecución; las versiones incompatibles se
excluyen del leaderboard con advertencia y quedan registradas en
`leaderboard_exclusions.json`. Un caché legacy o dañado en la ruta activa se
archiva antes de recalcular. También se endureció el caché del evaluador común;
las métricas y rutas históricas de A no fueron reescritas por los comandos boost.

`candidate_a_reference.json` protege las configuraciones y predicciones históricas
mediante identidad JSON/texto normalizada, para que cambiar de sistema no produzca
falsos cambios. Los hashes físicos originales siguen disponibles en los reportes
históricos. No se migraron esos reportes a otro esquema.

## Auditoría y features

La auditoría para decidir features usa únicamente enero–octubre:

- 100.600 filas, 23.900 clientes, 18.558 con al menos dos observaciones.
- Solo cambia `dias_ultima_interaccion`.
- Cambia alguna vez en 60,91% de los clientes y 78,44% de los que tienen historial.
- Cambia en 59,51% de las 76.700 transiciones observadas.
- Mediana de cambio absoluto: 30 días; percentil 90: 214 días.
- Todas las transiciones observadas en desarrollo son de un mes. Las pruebas
  incluyen huecos de calendario aunque los datos actuales no los tengan.

Se añaden representaciones independientes, sin reemplazar `history`:

1. `temporal_structure`: conteo previo, indicador sin historial, tiempo desde
   primera observación, distancia a la observación previa y cobertura de 3 meses.
2. `temporal_lags`: estructura más lag1/lag2 y diferencias de interacción.
3. `temporal_v2`: lo anterior más media, desviación poblacional, mínimo, máximo y
   diferencia respecto de la media previa.

Los lag se refieren a observaciones anteriores. Las ventanas de tres meses usan
el intervalo calendario `[t-3,t)`, excluyendo la fila actual. Como la clave
cliente-mes es única, hay como máximo tres filas en ese intervalo; se implementa
con desplazamientos por cliente y máscaras de distancia mensual. No se completa
historia usando el futuro. La ausencia de valores históricos se mantiene como NaN
y se acompaña de conteos/indicadores. El orden original se restaura por posición.

## Resultados agosto–octubre

| Modelo / representación | Agosto | Septiembre | Octubre | Media | Peor mes |
|---|---:|---:|---:|---:|---:|
| A: CatBoost 75% + HGB observed_time 25% | 0.264211 | 0.249065 | 0.268360 | **0.260545** | 0.249065 |
| HGB original | 0.259736 | 0.245847 | 0.264803 | 0.256795 | 0.245847 |
| HGB observed_time / temporal_structure | 0.258006 | 0.245825 | 0.269319 | 0.257716 | 0.245825 |
| HGB temporal_lags | 0.251962 | 0.251937 | 0.268117 | 0.257339 | 0.251937 |
| HGB temporal_v2 | 0.257057 | 0.246690 | 0.265415 | 0.256387 | 0.246690 |
| CatBoost original | 0.261989 | 0.247553 | 0.265506 | 0.258349 | 0.247553 |
| CatBoost temporal_lags | 0.259021 | 0.243923 | 0.263159 | 0.255367 | 0.243923 |
| CatBoost 75% + HGB temporal_lags 25% | 0.263276 | 0.250927 | 0.269514 | 0.261239 | 0.250927 |

A se reprodujo con el mismo Gini y diferencia máxima de probabilidades
aproximadamente 1e-15 frente al CSV histórico. La estructura general añade
información redundante en estas secuencias completas. Los rezagos mejoran
septiembre y el peor mes de HGB, pero empeoran agosto; las estadísticas adicionales
no mejoran la media. CatBoost con rezagos empeora los tres meses. La combinación
fija aporta una mejora pequeña, insuficiente para el criterio registrado.

## Diagnóstico de noviembre

`boost-diagnose` usa exclusivamente predicciones ya guardadas de A. No entrena
modelos, no calcula Gini de diciembre y no evalúa variantes de B. Publica tasas,
historial, cuantiles numéricos, categorías, scores y desempeño por riesgo/canal/
región. Exige 30 positivos y 30 negativos para reportar Gini por segmento.

El bootstrap exploratorio por cliente para noviembre menos media de desarrollo
da IC95% aproximadamente **[-0.06248, 0.00715]**, incluyendo cero. La composición
por historial no explica por sí sola la caída, y el grupo histórico también
empeora. Los datos no permiten atribuir causalmente el descenso a una sola causa.

## Ronda 2 — ajuste acotado de HGB

El presupuesto se registró antes de entrenar: dos configuraciones originales,
tres con `observed_time` y tres con `temporal_lags`. Solo cambian hojas máximas,
muestras mínimas por hoja y L2. Se mantienen 150 iteraciones, learning rate 0.06,
seed 42 y early stopping desactivado. El manifiesto impide cambiar o ampliar la
ronda ya registrada; una hipótesis nueva debe tener otro nombre.

La shortlist aplica el criterio de ganancia/estabilidad del protocolo contra la
referencia de las mismas features. Se escogen como máximo dos representaciones
distintas por media de Gini. Los pesos de A se conservan en las dos sustituciones.

| Modelo | Agosto | Septiembre | Octubre | Media | Peor mes |
|---|---:|---:|---:|---:|---:|
| HGB observed_time referencia | 0.258006 | 0.245825 | 0.269319 | 0.257716 | 0.245825 |
| HGB observed_time shallow | 0.264583 | 0.252397 | 0.268248 | **0.261742** | 0.252397 |
| HGB original shallow | 0.261345 | 0.253887 | 0.263969 | 0.259734 | 0.253887 |
| HGB temporal_lags shallow | 0.256875 | 0.252184 | 0.266427 | 0.258495 | 0.252184 |

El mejor HGB utiliza `max_leaf_nodes=7`, `min_samples_leaf=200` y
`l2_regularization=30`. Los ocho resultados y los grupos por historial se guardan
en `hgb_comparison.csv` y `comparison.json`.

| Solución completa | Media | Ganancia frente a A | Peor mes |
|---|---:|---:|---:|
| A original | 0.260545 | — | 0.249065 |
| CatBoost 75% + HGB observed_time shallow 25% | 0.260718 | +0.000173 | 0.249786 |
| CatBoost 75% + HGB original shallow 25% | 0.260368 | -0.000177 | 0.249753 |

Ninguna combinación alcanza el mínimo 0.002. No se amplió el presupuesto ni se
buscaron otros pesos. Una mejora individual no garantiza una mejora equivalente
al mezclar probabilidades con CatBoost.

`boost-hgb-stats` utiliza únicamente los artefactos vigentes de la ronda, verifica
su identidad y calcula incertidumbre del mejor HGB, sin entrenar ni consultar
noviembre. El código de este diagnóstico y el reporte de entrada se identifican
con hashes propios; no cambian la identidad de los entrenamientos.

- Mejor HGB menos A: ganancia observada **0.001197**, IC95% **[-0.004235, 0.006682]**.
- Mejor HGB menos su referencia: ganancia **0.004026**, IC95% **[-0.002159, 0.009726]**.
- 500 réplicas pareadas por cliente, todas válidas. Ambos intervalos incluyen cero.

Son diagnósticos exploratorios posteriores a selección; no prueban superioridad
independiente. El HGB shallow se conserva como candidato alternativo, pero A sigue
siendo la referencia de entrega. No se congeló un B ni se evaluó noviembre.

## Verificación y siguiente decisión

22 pruebas aprobadas: identidad CRLF/LF, cambios de código/datos/protocolo/spec,
integridad de predicciones, exclusión del leaderboard, protección de A, causalidad
temporal, ventanas con huecos, orden, independencia por cliente y métricas con
clases ausentes, presupuesto de ronda inmutable, shortlist estable y aislamiento
de cachés/reportes entre rondas. La ejecución real se verificó en Linux; se simulan formatos de
Windows en las pruebas, sin afirmar una ejecución nativa en Windows.

Una nueva ronda requiere una hipótesis distinta y un presupuesto registrado en
desarrollo. No ampliar ventanas o ensayos para forzar una mejora. La entrega sigue
basada en A; noviembre no se utiliza para rescatar variantes descartadas.
