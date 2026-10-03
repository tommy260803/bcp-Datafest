# Contexto de construcción — DataFest BCP–ESAN

Fecha del contexto: 3 de octubre de 2026. Zona horaria del equipo: America/Lima.

Este documento reúne las decisiones de la conversación y define la construcción por fases de un proyecto reproducible de machine learning. Está dirigido al equipo y al asistente de programación que trabajará en su entorno local. Léelo completo antes de implementar. Los hechos observados deben contrastarse con los archivos locales al iniciar; las decisiones metodológicas deben mantenerse consistentes durante los experimentos.

## 1. Objetivo final y prioridad del proyecto

Construir un proceso que aprenda de los datos históricos, estime la probabilidad de primera conversión de cada observación de diciembre de 2026 y produzca un archivo `outputs/submission.csv` válido para la competencia.

La métrica oficial es:

```text
Gini = 2 * ROC_AUC - 1
```

El objetivo técnico es conseguir el mejor ordenamiento predictivo que podamos justificar mediante validación temporal. No se conoce el Gini de otros equipos ni existe una puntuación que podamos garantizar.

El archivo final debe contener exactamente:

```csv
id_cliente,prediccion
```

Debe incluir las 9.900 observaciones de `test.csv`, mantener sus identificadores y su orden original, y contener probabilidades finitas entre 0 y 1. No debe incluir `mes`, índices de pandas ni columnas adicionales. Los valores de `sample_submission.csv` son ejemplos, no predicciones utilizables.

La imagen de instrucciones indica que cada equipo presenta una sola solución final mediante un único representante. Este proyecto prepara y verifica el archivo; el representante realizará la entrega al organizador. El equipo debe confirmar la hora y el mecanismo de entrega del 7 de octubre.

### Alcance acordado

El trabajo se concentra en preparación de datos, feature engineering, comparación de modelos, ajuste de hiperparámetros, evaluación de combinaciones, validación temporal, reproducibilidad y generación del CSV.

Quedaron fuera del alcance inicial la aplicación web, el dashboard, Streamlit, Plotly, SHAP y una presentación comercial. No son requisitos del archivo evaluado. Las métricas complementarias pueden guardarse en tablas internas si ayudan a diagnosticar el modelo.

El equipo tiene cinco programadores, cuatro días y acceso a asistentes de IA. Los asistentes pueden apoyar la implementación y revisión; la conservación de modelos o variables se decide con resultados medidos.

## 2. Qué significa la variable objetivo

Cada fila representa un cliente observado durante un mes.

- `objetivo = 0`: no ocurrió la primera conversión durante ese mes.
- `objetivo = 1`: ocurrió la primera conversión durante ese mes.
- Un cliente que registra `objetivo = 1` deja de aparecer en meses posteriores.

Ejemplo conceptual:

| id_cliente | mes | objetivo |
|---|---|---|
| 101 | 202601 | 0 |
| 101 | 202602 | 0 |
| 101 | 202603 | 1 |

El cliente 101 ya no aparece a partir de abril. Para predecir marzo podemos usar lo conocido en enero y febrero; su desaparición en abril no puede ser una variable predictora de marzo.

Se entrena con las filas etiquetadas tanto en 0 como en 1. No se eliminan los meses negativos ni se conserva únicamente la última fila de cada cliente. En diciembre se predicen todas las filas del test: no tener conversiones anteriores no significa que su probabilidad de diciembre sea cero.

Los datos contienen clientes con características y productos bancarios. La documentación no identifica la acción comercial específica que se considera conversión. No asumir que significa contratar un préstamo, seguro, tarjeta o convertirse por primera vez en cliente del banco.

La pregunta estadística es: «¿Qué probabilidad tiene este cliente de convertir en el mes evaluado, considerando que todavía no ha convertido y la información disponible para ese momento?».

La estructura admite una interpretación de primera conversión o supervivencia en tiempo discreto. Un clasificador cliente-mes es una forma viable de modelarla; no necesitamos imponer un modelo de supervivencia especializado.

## 3. Archivos y contrato de datos

Los archivos originales se ubicarán en `data/`. Si ya existe otra ubicación, detectarla, documentarla y mantener una configuración coherente; no crear copias divergentes.

| Archivo | Contenido esperado |
|---|---|
| `data/train.csv` | 110.100 filas y 25 columnas; enero–noviembre de 2026; incluye `objetivo`. |
| `data/test.csv` | 9.900 filas y 24 columnas; diciembre de 2026; no incluye `objetivo`. |
| `data/sample_submission.csv` | Estructura de entrega y una fila por observación del test. |
| `data/metaData.csv` | Diccionario de variables, tipos y notas. |

El separador es la coma, los decimales usan punto y las booleanas se representan mediante `True` y `False`. Leerlas explícitamente y comprobar sus valores: no convertir cadenas a booleano mediante reglas que interpreten cualquier texto como verdadero.

Los meses usan el entero `AAAAMM`. Son fechas del escenario de la competencia; no deben cambiarse por comparación con la fecha real del equipo.

### Identificación y tiempo

| Variable | Uso |
|---|---|
| `id_cliente` | Identificar, agrupar historial y reconstruir la entrega. Excluirlo como predictor directo. |
| `mes` | Orden temporal, definición de cortes y cálculo causal de historia. La referencia inicial no lo usa como predictor; cualquier uso predictivo adicional debe evaluarse explícitamente. |

### Predictores numéricos

| Variable | Significado |
|---|---|
| `edad` | Edad del cliente. |
| `ingresos` | Ingresos anuales. |
| `ratio_deuda_ingresos` | Proporción de deuda sobre ingresos. |
| `antiguedad_cuenta_meses` | Antigüedad de la cuenta en meses. |
| `numero_productos` | Cantidad de productos bancarios activos. |
| `saldo_promedio` | Saldo promedio de la cuenta. |
| `dias_ultima_transaccion` | Días desde la transacción más reciente. |
| `antiguedad_direccion_meses` | Antigüedad de la dirección registrada. |
| `visitas_web_ultimos_90_dias` | Visitas web durante los últimos 90 días. |
| `distancia_sucursal_km` | Distancia aproximada a la sucursal más cercana. |
| `dia_preferido_pago` | Día preferido del mes para realizar pagos. |
| `dias_ultima_interaccion` | Días desde la interacción más reciente registrada. |

### Predictores categóricos

`ocupacion`, `region`, `canal_adquisicion`, `banda_riesgo`, `dispositivo_principal`.

### Predictores booleanos

`tiene_tarjeta_credito`, `activo_movil`, `es_nuevo_cliente`, `tiene_prestamo`, `tiene_seguro`.

### Etiqueta y salida

`objetivo` es la etiqueta binaria disponible en entrenamiento. `prediccion` es la probabilidad que se exporta. Son campos distintos.

La moneda de ingresos y saldo no se especifica en el diccionario. No interpretar estos importes automáticamente como soles ni atribuirles márgenes, rentabilidad o capacidad de pago que no estén documentados. `canal_adquisicion` no indica el canal óptimo de contacto.

## 4. Hallazgos de la exploración previa

Se procesaron las filas completas de los archivos adjuntos durante la conversación. Estos resultados son referencias para la auditoría local, no sustituyen la comprobación de los archivos que reciba el proyecto.

| Hallazgo | Resultado observado |
|---|---|
| Clientes distintos en entrenamiento | 24.628. |
| Filas positivas en entrenamiento | 16.567. |
| Tasa agregada de conversión por cliente-mes | 15,0472 %. No es la proporción de clientes que convierten alguna vez. |
| Faltantes en train y test | No se encontraron. |
| Duplicados de la clave cliente-mes | No se encontraron. |
| Más de una conversión positiva por cliente | No se encontraron casos. |
| Filas posteriores a la conversión de ese cliente | No se encontraron. |
| Clientes del test presentes en entrenamiento | 8.061, equivalentes al 81,4242 % del test. |
| Clientes del test sin observaciones anteriores | 1.839, equivalentes al 18,5758 % del test. |
| Clientes de noviembre sin observaciones antes de noviembre | 728 de 9.500; 7,6632 %. |
| Predictores que cambian dentro de un mismo cliente | Solo `dias_ultima_interaccion`; los otros 21 predictores permanecen fijos en estos archivos. |

Consecuencias para la implementación:

1. No asumir que podemos crear tendencias informativas de ingresos, saldo, productos o visitas: sus diferencias temporales son cero en estos datos.
2. `dias_ultima_interaccion` y `dias_ultima_transaccion` son campos distintos. El primero cambia; el segundo permanece fijo por cliente en los archivos examinados.
3. «Sin historial» se calcula por existencia de registros anteriores. No equivale a `es_nuevo_cliente`.
4. El tiempo desde la primera observación no equivale a la antigüedad real de la relación bancaria, ni al número de campañas recibidas.
5. La población de diciembre contiene una mayor proporción de clientes sin historial que noviembre. Debemos comprobar su efecto en la evaluación.

Las asociaciones exploratorias más visibles fueron riesgo, cantidad de productos, recencia transaccional y actividad móvil. Son candidatas para el modelado, no reglas causales ni garantías de importancia después de controlar otras variables.

## 5. Pruebas preliminares ya realizadas

Durante la conversación se ejecutó una comparación rápida con scikit-learn. No se entrenaron CatBoost ni LightGBM y no se hizo una búsqueda sistemática con Optuna. No existe todavía una entrega final de diciembre.

Los scripts y modelos temporales de aquella sesión no deben suponerse disponibles en la computadora del equipo. Hay que crear el proyecto y reproducir las pruebas. Las diferencias de versiones o detalles de implementación pueden cambiar las cifras; no ajustar código para forzar una coincidencia artificial.

### Protocolo utilizado

- Prueba de septiembre: entrenamiento enero–agosto; evaluación en septiembre.
- Prueba de octubre: entrenamiento enero–septiembre; evaluación en octubre.
- Predictores originales: todas las columnas salvo `id_cliente`, `mes` y `objetivo`.
- Los transformadores se ajustaron exclusivamente con la porción de entrenamiento de cada corte.
- No se evaluó el Gini de modelos sobre noviembre. Sí se inspeccionaron conteos y tasas descriptivas de ese mes durante la exploración; no describirlo como un mes cuyos datos nunca se observaron.

| Variante | Gini septiembre | Gini octubre |
|---|---:|---:|
| Regresión logística | 0,234561 | 0,239216 |
| Regresión logística con curvas spline | 0,228915 | 0,244943 |
| HistGradientBoosting con variables originales | 0,245847 | 0,264803 |
| HistGradientBoosting con variables originales e historial | 0,239582 | 0,260359 |
| Regresión logística con curvas spline e historial | 0,226233 | 0,243092 |

La ventaja del boosting frente a la regresión logística fue más clara en octubre. Las pequeñas diferencias entre boosting con y sin historial no permiten concluir que el historial siempre perjudicará: los intervalos aproximados obtenidos mediante remuestreo incluían cero. Debe probarse como candidato.

### Configuraciones de referencia

Regresión logística:

- Numéricas: `StandardScaler` ajustado en entrenamiento.
- Booleanas: paso directo como variables binarias.
- Categóricas: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Modelo: `C=0.1`, `max_iter=400`, `solver="lbfgs"`.

Variante con curvas: antes de escalar las numéricas se usó `SplineTransformer(n_knots=4, degree=2, knots="quantile", extrapolation="linear")`.

HistGradientBoosting:

```json
{
  "max_iter": 150,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 100,
  "l2_regularization": 10,
  "learning_rate": 0.06,
  "early_stopping": false,
  "random_state": 42
}
```

Las categóricas se codificaron con `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` y se declararon como categóricas para el estimador. Las numéricas y booleanas se pasaron sin escalar a los árboles.

El bloque histórico probado contenía:

- `observaciones_previas`: conteo acumulado de filas anteriores del cliente.
- `sin_historial`: indicador de conteo previo igual a cero.
- `interaccion_previa`: valor de `dias_ultima_interaccion` de la observación anterior.
- `cambio_interaccion`: valor actual menos valor anterior.
- `media_interaccion_previa`: media de hasta tres observaciones anteriores; primero desplazamiento de una fila, después ventana móvil.

Para la primera observación se completaron la interacción previa y la media previa con el valor actual, la diferencia con cero y se conservó el indicador `sin_historial`. Este es un tratamiento del experimento de referencia; pueden compararse otras alternativas dentro del proceso correcto de validación.

Como diagnóstico complementario, el boosting original capturó 458 de 1.393 conversiones en septiembre al seleccionar 1.980 clientes, y 559 de 1.628 conversiones en octubre al seleccionar 2.080 clientes. Eso representa seleccionar el 20 % de cada mes y capturar respectivamente 32,88 % y 34,34 % de sus conversiones observadas. No mide conversiones causadas por una campaña.

## 6. Tecnologías acordadas

| Tecnología | Responsabilidad |
|---|---|
| Python 3.12, CPython de 64 bits | Lenguaje y entorno objetivo del proyecto. |
| pandas y NumPy | Lectura, auditoría, agrupación temporal, creación de variables y exportación. |
| scikit-learn | Referencias, preprocesamiento, métricas AUC/Gini y herramientas de modelado. |
| CatBoost | Candidato principal de árboles con manejo de categorías. |
| LightGBM | Candidato principal de árboles y comparación con CatBoost. |
| Optuna | Búsqueda controlada de hiperparámetros. |
| uv | Entorno aislado y dependencias reproducibles mediante `pyproject.toml` y `uv.lock`. |
| Git y repositorio privado de GitHub | Control de versiones y colaboración. |
| VS Code y Jupyter | Desarrollo y exploración. Los pasos oficiales se ejecutan mediante scripts o comandos reproducibles. |
| pytest | Pruebas concretas de temporalidad, alineación y entrega. |

Utilizar formatos nativos de CatBoost/LightGBM y, cuando corresponda, `joblib` para conservar transformaciones y modelos propios de scikit-learn. Guardar configuraciones y métricas en JSON/CSV.

Las versiones exactas de bibliotecas se fijarán después de verificar una instalación compatible. No asumir que el entorno anterior de la conversación coincide con el entorno local. Si Python 3.12 no es viable en una máquina, documentar y acordar una alternativa compatible común antes de generar entornos divergentes.

Iniciar por CPU. Medir tiempos y memoria antes de ampliar la búsqueda. Controlar tanto la cantidad de ensayos simultáneos como los hilos internos de cada modelo para no saturar las computadoras. La GPU es una posibilidad de aceleración, no un requisito del proyecto.

Para estudios de Optuna que deban reanudarse, puede usarse SQLite local por integrante o estudio. No compartir el mismo archivo SQLite entre varias computadoras mediante una carpeta sincronizada. Consolidar resultados mediante exportaciones identificadas.

Los CSV originales, modelos pesados, entornos, credenciales y archivos temporales tendrán un tratamiento explícito en `.gitignore`. Cada integrante debe disponer de la misma versión de los datos por el medio autorizado para el equipo. El código, configuraciones, documentación y reportes compactos sí deben poder compartirse y revisarse.

## 7. Arquitectura mínima del proyecto

Trabajar con rutas relativas a la raíz y una ubicación de datos configurable. Si el repositorio ya contiene código, inspeccionarlo antes de cambiar su estructura.

| Ruta sugerida | Contenido |
|---|---|
| `CONTEXTO_BCP_DATAFEST.md` | Este documento. |
| `README.md` | Instalación, comandos y reproducción del resultado. |
| `ESTADO.md` | Fase actual, ejecuciones, decisiones, problemas y siguiente paso. |
| `pyproject.toml`, `uv.lock` | Entorno y dependencias. |
| `data/` | Cuatro CSV originales, conservados sin modificaciones. |
| `configs/` | Cortes, semillas, columnas, familias de variables y espacios de búsqueda. |
| `src/bcp_datafest/data.py` | Lectura y validación del esquema. |
| `src/bcp_datafest/features.py` | Variables originales y derivadas; historia calculada hacia atrás. |
| `src/bcp_datafest/splits.py` | Cortes por meses completos e identificación de historial por corte. |
| `src/bcp_datafest/models.py` | Construcción y preparación específica de cada familia de modelos. |
| `src/bcp_datafest/metrics.py` | AUC, Gini y diagnósticos complementarios. |
| `src/bcp_datafest/tuning.py` | Estudios de Optuna y registro de resultados. |
| `src/bcp_datafest/ensemble.py` | Comparación de combinaciones de probabilidades. |
| `src/bcp_datafest/predict.py` | Reentrenamiento final, predicción y alineación. |
| `src/bcp_datafest/submission.py` | Exportación y validación del CSV. |
| `src/bcp_datafest/__main__.py` | Entrada común por línea de comandos. |
| `tests/` | Comprobaciones automáticas que protegen riesgos concretos. |
| `notebooks/` | Exploraciones opcionales que no sustituyen al proceso ejecutable. |
| `artifacts/runs/<run_id>/` | Configuración, métricas, predicciones de validación y detalles del ensayo. |
| `artifacts/models/final/` | Modelo seleccionado, transformaciones, columnas y metadatos de entrenamiento. |
| `reports/` | Auditoría, comparación de candidatos y evaluación final. |
| `outputs/` | Candidatos de entrega y archivo final. |

Implementar una entrada simple. El siguiente es un contrato de comandos propuesto; no debe suponerse que ya funciona antes de construir el paquete:

```bash
uv sync
uv run python -m bcp_datafest audit
uv run python -m bcp_datafest baseline
uv run python -m bcp_datafest experiment --config configs/experiments.json
uv run python -m bcp_datafest tune --model catboost
uv run python -m bcp_datafest tune --model lightgbm
uv run python -m bcp_datafest blend
uv run python -m bcp_datafest evaluate-final
uv run python -m bcp_datafest fit-final
uv run python -m bcp_datafest predict --output outputs/submission.csv
uv run python -m bcp_datafest validate-submission --file outputs/submission.csv
```

Puede simplificarse el diseño si mantiene las responsabilidades y la reproducción completa. Evitar agregar servicios o infraestructura que no sean necesarios para entrenar y generar el CSV.

## 8. Validación y selección de modelos

### 8.1 Cortes temporales

Todos los integrantes deben usar los mismos cortes, semillas de referencia y definiciones de métricas. Las particiones se harán por meses completos, no por posiciones arbitrarias de filas.

| Uso | Entrenamiento | Evaluación |
|---|---|---|
| Desarrollo adicional | Enero–julio | Agosto |
| Desarrollo y comparación con referencias | Enero–agosto | Septiembre |
| Desarrollo y selección | Enero–septiembre | Octubre |
| Evaluación final de la configuración elegida | Enero–octubre | Noviembre |
| Entrenamiento para entrega | Enero–noviembre | Predicción de diciembre, sin etiquetas conocidas |

Los resultados previos corresponden solo a septiembre y octubre. El protocolo de implementación propuesto añade agosto como tercer mes de desarrollo. Si el presupuesto exige usar únicamente septiembre y octubre, decidirlo y registrarlo antes de la búsqueda amplia, aplicándolo igual a todos los candidatos.

La separación aleatoria de filas no será el criterio de selección. El mismo cliente puede estar en entrenamiento y en un mes de validación posterior: esto reproduce la predicción de clientes que tienen historial. También se evaluarán los clientes que aparecen por primera vez en cada corte.

El `early stopping` interno no debe crear silenciosamente una partición aleatoria para elegir la versión final. En desarrollo, usar una evaluación temporal explícita cuando la biblioteca lo permita. En referencias con un número fijo de iteraciones, desactivar el `early_stopping` aleatorio.

### 8.2 Métrica principal

Calcular ROC AUC a partir de la probabilidad de clase 1 y derivar Gini. No usar etiquetas binarias ni un umbral de 0,5 para esta métrica. La exactitud, F1 y el umbral comercial no son el objetivo de la selección.

La referencia para comparar candidatos será la media de los Gini de los meses de desarrollo, dando inicialmente el mismo peso a cada mes. Mostrar además cada mes individual y el peor mes. No sustituir esa media por el AUC de todas las predicciones concatenadas sin explicar que son evaluaciones diferentes.

Elegir el criterio antes de la búsqueda. Considerar estabilidad, simplicidad y costo cuando las diferencias sean pequeñas. Si una mejora dudosa determina una decisión importante, estimar su incertidumbre mediante remuestreo pareado: las predicciones comparadas deben usar los mismos ejemplos. Al reunir meses, remuestrear por cliente para respetar la repetición de observaciones.

### 8.3 Clientes con y sin historial

En cada corte, definir «con historial» usando exclusivamente la existencia del cliente en meses anteriores al mes evaluado. Medir tamaño, positivos y Gini de ambos grupos. Si un grupo no contiene ambas clases, reportar que su AUC/Gini no está definido; no inventar cero ni omitirlo silenciosamente.

Como diagnóstico secundario, se puede aproximar la composición de diciembre ponderando las observaciones de validación. Con `q = 1839/9900` como fracción observada sin historial en el test y `r` como fracción de ese grupo en la validación:

```text
peso_sin_historial = q / r
peso_con_historial = (1 - q) / (1 - r)
Gini_ponderado = 2 * AUC(y, probabilidad, sample_weight=pesos) - 1
```

Recalcular `q` de los archivos locales. Si algún grupo falta en la validación, no aplicar esta fórmula; reportar la limitación. Se calcula un AUC global ponderado, no un promedio de AUC por grupos: el ordenamiento entre los grupos también cuenta.

Esta ponderación utiliza características observables del test, no etiquetas de diciembre. No corrige automáticamente cambios de distribución dentro de cada grupo. Mantener el Gini temporal sin ponderar como métrica principal y la evaluación ponderada como diagnóstico acordado antes de observar la evaluación final. Respetar cualquier restricción adicional de las bases sobre uso del test.

### 8.4 Noviembre reservado para la evaluación final

La familia de modelo, variables, hiperparámetros, criterio de selección, pesos de combinación y política de iteraciones deben estar congelados antes de calcular el Gini final de noviembre.

No utilizar noviembre para `early stopping`, selección de variables, calibración, ajuste de pesos ni búsquedas repetidas. Para entrenar hasta octubre y evaluar noviembre, fijar las iteraciones con una regla definida a partir de desarrollo, como la mediana de las mejores iteraciones temporales, o una configuración fija previamente seleccionada.

Registrar una evaluación final de la configuración congelada. Si se descubre un error real de datos o implementación, corregirlo y dejar constancia; las evaluaciones posteriores ya no deben presentarse como una primera prueba completamente independiente. Un resultado menor al esperado, por sí solo, no autoriza ajustar reiteradamente sobre noviembre.

Después de esta evaluación se permite entrenar el modelo seleccionado con enero–noviembre para producir diciembre. Las etiquetas de diciembre no existen en los archivos disponibles, así que su Gini solo lo calcula el organizador.

## 9. Feature engineering y prevención de fugas

### 9.1 Bloques a comparar

| Bloque | Variables propuestas |
|---|---|
| Original | Los 22 predictores originales con preparación apropiada al modelo. |
| Relación económica | Ingreso mensual estimado `ingresos/12`; saldo relativo a ese ingreso, con control de denominadores. No llamarlo capacidad de pago. |
| Vinculación | Combinaciones de tarjeta, préstamo y seguro; interacción con `numero_productos` y riesgo. No suponer que la suma de los tres indicadores coincide con `numero_productos`. |
| Actividad | Interacciones entre actividad móvil, recencia transaccional y productos; relaciones no lineales cuando corresponda. |
| Historial | Conteo previo, indicador de ausencia de historial, rezago de interacción, diferencia y media o mediana de observaciones previas. |
| Tiempo observado | Tiempo transcurrido desde la primera aparición usando exclusivamente la historia disponible. Diferenciarlo del número de observaciones. |

Probar bloques mediante comparación controlada: mismo algoritmo, corte y presupuesto, cambiando el bloque investigado. Después ajustar los candidatos prometedores. Una mayor cantidad de variables no es evidencia de mejora.

Agrupar registros de un mismo cliente para construir historia no es clustering. Agrupar clientes similares puede explorarse más adelante si existe una hipótesis concreta y presupuesto; no es un componente obligatorio. Tampoco hay relaciones entre clientes para justificar un grafo como parte del alcance inicial.

### 9.2 Reglas de temporalidad

Para una fila del mes `t`, cualquier agregado histórico debe usar únicamente filas anteriores a `t`. Ordenar por cliente y fecha, desplazar con `shift(1)` y luego calcular la ventana correspondiente. No restar enteros `AAAAMM` como si fueran distancias uniformes entre meses; utilizar fechas o índices mensuales.

Si hay huecos temporales, distinguir «observación anterior» de «mes anterior». Conservar indicadores de falta de historia cuando sean necesarios. No rellenar historia mediante valores de meses futuros.

Los valores originales del mes predicho se usan bajo el supuesto de la competencia de que estaban disponibles al momento de generar la predicción. El corte exacto respecto de la conversión no está documentado: registrar ese supuesto y plantear la pregunta al organizador, sin inventar un significado distinto para las columnas.

### 9.3 Información que no puede transformarse en predictor

- Desaparición futura del cliente, fecha de su última aparición en todo el dataset o cantidad total de meses que acabará apareciendo.
- Saber si un cliente de un mes anterior llegará a diciembre o pertenecerá al test.
- Etiqueta del propio registro de validación, conversiones futuras o estadísticas calculadas con ellas.
- Identificadores numéricos usados como sustitutos de perfil, cohorte o etiqueta.
- Transformaciones, selección de variables, imputaciones o codificaciones aprendidas con la validación o el test.

Las conversiones históricas positivas por cliente no aportan una señal usable en esta población: antes de la primera conversión todas sus etiquetas previas son cero y después deja de aparecer.

Las ventanas de 90 días suministradas pueden solaparse entre meses. El solapamiento no es automáticamente fuga si cada valor existía en el corte de predicción; tampoco justifica mezclar el futuro al construir variables.

### 9.4 Preparación específica del modelo

Para regresión, ajustar escaladores y codificadores dentro de cada corte. Para CatBoost, declarar las categóricas y utilizar su tratamiento nativo. Para LightGBM, garantizar una codificación/categoría consistente y un tratamiento explícito de categorías nuevas. Todo preprocesamiento aprendido se ajusta en entrenamiento y se conserva para inferencia.

No aplicar sobremuestreo, SMOTE, balanceo o pesos de clase de forma automática por la tasa de positivos. Son hipótesis adicionales que deben justificar su uso; una configuración inicial sin ellos es una referencia válida.

## 10. Modelos, tuneo y combinaciones

### Referencias

Reproducir primero una regresión logística y HistGradientBoosting. Esto permite comprobar la infraestructura y tener una referencia antes de invertir en búsquedas.

### Candidatos principales

Comparar `CatBoostClassifier` y `LGBMClassifier` para clasificación binaria probabilística. La evaluación oficial es un ranking, pero no exige entrenar un modelo de ranking por pares ni construir grupos artificiales.

Optuna ajustará un espacio acotado y documentado: complejidad de árboles, tasa de aprendizaje, cantidad de iteraciones, regularización, tamaño mínimo de hojas y muestreo cuando corresponda. Comprobar que los parámetros de cada biblioteca sean compatibles entre sí y estén realmente activos.

Usar TPE como punto de partida de búsqueda y un presupuesto explícito por familia. Empezar, por ejemplo, con 20–30 ensayos para medir costo y utilidad, y ampliar a 50–100 si hay mejora y tiempo. Son presupuestos iniciales de implementación, no un requisito de completar cierta cantidad. Toda comparación debe registrar también el presupuesto consumido.

El objetivo de cada ensayo debe ser la métrica temporal acordada, no el resultado de entrenamiento. Si se usa poda de ensayos, comparar etapas equivalentes y no favorecer a un modelo por evaluar solo el mes más fácil. Mantener un registro de ensayos fallidos y su causa.

### Combinación de predicciones

Conservar candidatos individuales fuertes y comparar promedios ponderados de sus probabilidades sobre las mismas observaciones de validación:

```text
p_final = w1*p_modelo1 + w2*p_modelo2 + ...
wi >= 0
suma(wi) = 1
```

Usar predicciones de meses que cada modelo no empleó para entrenar. Elegir pesos únicamente con desarrollo y comparar el conjunto contra el mejor modelo individual. Probar pocas combinaciones justificadas para limitar el sobreajuste a los meses de desarrollo.

El promedio de varias semillas también puede probarse si aporta estabilidad. No imponer un ensamble si una solución individual funciona mejor. Evitar stacking con predicciones de entrenamiento: requeriría una validación temporal adicional correctamente diseñada.

Como Gini evalúa ordenamiento, la calibración no es una prioridad independiente. Si se usa para compatibilizar probabilidades de una combinación, ajustarla solo con desarrollo y comprobar el efecto sobre Gini. No convertir percentiles de ranking en supuestas probabilidades reales ni binarizar el resultado.

## 11. Construcción por fases

La siguiente secuencia es el trabajo que debe ejecutar el asistente local. Avanzar al cumplir el criterio de cierre de cada fase, registrar los resultados y comunicar el siguiente paso. No detenerse para pedir aprobación rutinaria entre fases; si faltan datos o acceso indispensable, explicar el bloqueo concreto y continuar el trabajo independiente que sea posible.

### Fase 0. Preparación del proyecto

Inspeccionar la carpeta, instrucciones locales y código existente; localizar los cuatro CSV; verificar Python, uv y recursos disponibles. Crear o adaptar el proyecto, entorno, dependencias, configuraciones, documentación y exclusiones de Git. Preservar el trabajo existente del equipo.

Registrar el horizonte de cuatro días y confirmar la hora límite cuando el equipo la tenga. Medir recursos antes de iniciar entrenamiento intensivo.

**Entregables:** entorno instalable, entrada ejecutable, rutas configuradas, `README.md` y `ESTADO.md` iniciales.

**Cierre:** importar las bibliotecas necesarias, localizar los datos y ejecutar un comando de auditoría reproducible.

### Fase 1. Auditoría reproducible

Comprobar esquema, tipos, filas, claves, meses, faltantes, etiqueta binaria, positivos por mes, clientes repetidos, regla de primera conversión y correspondencia con la plantilla de entrega. Medir variación intracliente y la proporción sin historial por mes. Registrar hashes de los cuatro archivos para que todos trabajen sobre los mismos datos.

No cambiar los datos originales para que coincidan con el resumen previo. Si existe una diferencia, documentar cuál es la fuente local y resolverla antes de usar resultados comparativos.

**Entregable:** `reports/audit.md` y un resumen estructurado de la auditoría.

**Controles de aceptación:**

- [ ] Archivos y esquema compatibles con el contrato de la competencia.
- [ ] Claves y orden de entrega identificados; ausencia de duplicados confirmada o incidencia documentada.
- [ ] Etiquetas y regla de primera conversión verificadas.
- [ ] Cortes mensuales y definición de cliente sin historial comprobados.

### Fase 2. Referencias y validación común

Implementar los cortes compartidos, métricas, transformaciones por corte y modelos de referencia. Ejecutar septiembre y octubre para comparar con el experimento previo; incorporar agosto al protocolo de desarrollo si el presupuesto lo permite, antes de la búsqueda amplia.

Guardar predicciones de validación con identificador, mes, etiqueta real, probabilidad, corte y nombre del modelo. Registrar resultados globales y por disponibilidad de historia. Puede generarse un primer CSV de prueba de formato desde la referencia; etiquetarlo como candidato, no como entrega final seleccionada.

**Entregables:** referencias reproducibles, predicciones fuera de entrenamiento y primer reporte comparativo.

**Cierre:** todas las familias pueden evaluarse con las mismas observaciones, métricas y definición temporal.

### Fase 3. Variables derivadas

Implementar los bloques de variables por separado. Comparar original, relaciones económicas, vinculación, actividad e historial. Verificar que los campos estáticos no generen supuestas tendencias. Tratar adecuadamente clientes sin antecedentes y restaurar el orden cuando se ordenen filas para calcular historia.

Construir pruebas focalizadas: cambiar o agregar filas futuras no debe modificar variables históricas ya calculadas; la primera observación debe tener historial ausente; ordenar datos para agruparlos no debe cambiar la correspondencia final entre cliente y predicción.

**Entregables:** funciones de variables, configuración de bloques y tabla de comparaciones controladas.

**Cierre:** conservar candidatos por evidencia temporal, con las pruebas de causalidad de historia y alineación aprobadas.

### Fase 4. Entrenamiento competitivo y tuneo

Entrenar CatBoost y LightGBM con la referencia original y los bloques prometedores. Ejecutar estudios acotados de Optuna con los cortes comunes, registrar iteraciones elegidas en desarrollo y revisar sobreajuste entre meses. Mantener candidatos alternativos de regresión o HistGradientBoosting cuando compitan o aporten errores distintos.

El entrenamiento y tuneo pueden repartirse entre integrantes. No cambiar la definición de una variable o corte a mitad de una comparación sin versionar el cambio.

**Entregables:** estudios, configuraciones candidatas, tiempos, métricas por mes y predicciones comparables.

**Cierre:** lista corta de candidatos con resultados trazables y un presupuesto de entrenamiento compatible con la entrega.

### Fase 5. Selección y combinación

Comparar modelos individuales, combinaciones de probabilidades y, si procede, promedio de semillas. Revisar desempeño por disponibilidad de historial y el diagnóstico de composición de diciembre. Si una diferencia pequeña cambia la elección, analizar su incertidumbre en lugar de asumir que cualquier decimal representa una mejora estable.

Elegir la solución y congelar columnas, bloques, hiperparámetros, pesos, semillas, transformaciones y regla de número de iteraciones. Guardar la configuración elegida antes de evaluar noviembre.

**Entregables:** `reports/model_selection.md` y una configuración final bloqueada para la evaluación.

**Cierre:** existe una decisión reproducible basada exclusivamente en los meses de desarrollo.

### Fase 6. Evaluación final en noviembre

Entrenar la configuración congelada con enero–octubre y predecir noviembre. Evaluar Gini global, grupos con/sin historial y diagnósticos acordados. Comparar la magnitud con desarrollo y explicar límites; no abrir una búsqueda de hiperparámetros sobre noviembre.

**Entregable:** `reports/final_validation.md`, con configuración, datos utilizados, resultados y cualquier incidencia real.

**Controles de aceptación:**

- [ ] Configuración guardada antes de consultar el Gini de noviembre.
- [ ] Noviembre no usado para elegir iteraciones, variables, pesos ni calibración.
- [ ] Métricas calculadas con probabilidades de clase 1 y claves alineadas.
- [ ] Resultado y limitaciones registrados sin convertir una evaluación repetida en una supuesta prueba independiente.

### Fase 7. Reentrenamiento y predicción de diciembre

Reentrenar la solución elegida con enero–noviembre y las iteraciones fijadas a partir de desarrollo. Ajustar los transformadores finales solo con ese entrenamiento. Construir para diciembre las variables históricas permitidas usando la información disponible hasta noviembre y los predictores actuales suministrados en el test.

Guardar el modelo completo y sus metadatos. Generar una probabilidad para cada fila de diciembre y restaurar explícitamente el orden original del test usando una referencia de orden conservada desde la lectura.

**Entregables:** modelos finales, esquema de variables, configuración, probabilidades y `outputs/submission.csv`.

**Cierre:** todas las observaciones tienen una predicción válida y el archivo se puede regenerar a partir del código y la configuración registrada.

### Fase 8. Comprobación de entrega y documentación

Volver a leer el CSV exportado y verificar el contenido que efectivamente se entregará. Guardar un reporte de validación y hash del archivo. Evitar redondear agresivamente las probabilidades: conservar precisión suficiente, por ejemplo 10–15 cifras significativas, para no crear empates innecesarios.

**Checklist final de entrega:**

- [ ] 9.900 filas de datos y encabezado, concordantes con el test local validado.
- [ ] Columnas exactas y en este orden: `id_cliente,prediccion`.
- [ ] Identificadores idénticos a los de `test.csv`, fila por fila.
- [ ] Ninguna columna de índice, `mes`, etiqueta ni dato adicional.
- [ ] Todas las predicciones numéricas, finitas y comprendidas entre 0 y 1.
- [ ] Sin predicciones faltantes ni filas perdidas, duplicadas o reordenadas.
- [ ] Separador coma, punto decimal y codificación UTF-8.
- [ ] Predicciones producidas por el modelo elegido; no valores de ejemplo de la plantilla ni etiquetas binarias derivadas de un umbral.
- [ ] Se conserva la configuración, el modelo, los hashes de datos, la versión del código y el reporte final de Gini de noviembre.
- [ ] El representante sabe cuál es el único archivo final y el equipo ha confirmado la hora y vía de entrega.

**Entregables:** CSV verificado, reporte de comprobación y pasos de reproducción en `README.md`.

**Cierre:** archivo listo para que el representante lo presente. El Gini oficial de diciembre sigue siendo desconocido hasta la evaluación del organizador.

## 12. Reparto del trabajo entre cinco programadores

| Integrante | Responsabilidad principal | Acuerdo de integración |
|---|---|---|
| 1 | Auditoría y feature engineering. | Publica funciones y definiciones versionadas; conserva claves y orden. |
| 2 | CatBoost y su tuneo. | Usa los mismos cortes y exporta métricas y predicciones con el formato común. |
| 3 | LightGBM y alternativas. | Mantiene comparabilidad de datos, presupuesto y métricas. |
| 4 | Validación, selección y combinaciones. | Mantiene el protocolo y custodia el uso de noviembre; consolida candidatos. |
| 5 | Integración, reproducción y entrega. | Mantiene comandos, entorno, pruebas de archivo y documentación. |

Estos son roles para los integrantes humanos; pueden redistribuirse según experiencia. Utilizar ramas separadas y revisar cambios en los módulos compartidos. El responsable de integración no debe convertirse en desarrollador de una interfaz fuera de alcance.

Cada ejecución debe guardar, como mínimo: identificador de ensayo, modelo, versión del código, hashes de datos, configuración de variables, cortes, semillas, parámetros, iteraciones, versiones de bibliotecas, tiempos y Gini por mes. Mantener las predicciones de validación alineadas para poder probar combinaciones sin reentrenar innecesariamente.

## 13. Distribución de los cuatro días

| Día | Resultado esperado |
|---|---|
| Día 1: 3 de octubre | Entorno común, auditoría, referencias, cortes y proceso capaz de producir un CSV candidato. |
| Día 2: 4 de octubre | Comparaciones de variables y modelos; búsquedas de CatBoost y LightGBM en paralelo. |
| Día 3: 5 de octubre | Selección, evaluación de combinaciones, configuración congelada y evaluación final en noviembre. |
| Día 4: 6 de octubre | Reentrenamiento, CSV definitivo, reproducción y revisión de entrega. |

El 7 de octubre queda como fecha de entrega indicada por el usuario, con hora pendiente de confirmar. Dejar el archivo listo el día 6 evita depender de esa información para completar el desarrollo. Las actividades se ejecutan en paralelo según sus dependencias; no se suman como si una persona tuviera que hacer todo secuencialmente.

Si falta tiempo, conservar el mejor modelo validado y completar la entrega. La evaluación temporal y la alineación del CSV son requisitos; la cantidad de variables, ensayos o familias exploradas es ajustable.

## 14. Límites e información pendiente

El evento comercial exacto y el instante de medición de los predictores no están especificados. Mantener el nombre «primera conversión» y documentar el supuesto de disponibilidad de las variables. La falta del nombre del producto no impide aprender de la etiqueta entregada.

Los archivos no contienen tratamiento/control, costos de contacto, margen por conversión ni relaciones entre clientes. No presentar resultados de propensión como efectos causales de campañas, ganancias monetarias comprobadas o decisiones de aprobación crediticia.

No hay acceso al resultado real de diciembre. No fabricar un Gini de test ni usar la plantilla como si tuviera etiquetas. Una buena validación aumenta la confianza en el procedimiento, pero no garantiza ganar la competencia.

Por ahora no están confirmados los recursos de hardware de las cinco máquinas, la hora exacta de entrega ni la ubicación del repositorio. Resolver lo necesario durante la preparación sin detener tareas que pueden avanzar con los CSV locales.

## 15. Estado inicial para el asistente que recibe este documento

| Elemento | Estado al redactar este contexto |
|---|---|
| Objetivo y formato de entrega | Definidos. |
| Alcance | Concentrado en CSV y Gini. |
| Tecnologías | Definidas a nivel de bibliotecas; versiones exactas pendientes de instalación. |
| Auditoría previa | Realizada en la conversación; reproducir en el proyecto local. |
| Ensayos preliminares | Cinco variantes de scikit-learn, documentadas en la sección 5. |
| Código productivo y repositorio local | No se ha confirmado su existencia. Inspeccionar antes de crear. |
| CatBoost y LightGBM | Pendientes de entrenar y comparar. |
| Optuna y combinaciones | Pendientes. |
| Evaluación de modelos en noviembre | Pendiente. |
| Modelo final y CSV de diciembre | Pendientes. |

Comienza en la fase 0. No marques tareas como completadas porque este documento describa cómo realizarlas. Usa resultados verificables y conserva lo que ya exista si el usuario ha avanzado por su cuenta.

Al terminar cada bloque de trabajo, actualizar `ESTADO.md` con la fase, comandos ejecutados, resultados reales, archivos creados, decisiones, errores pendientes y siguiente acción. Mantener este contexto como referencia de alcance y método; cambiarlo si el usuario modifica una decisión e indicar el cambio.

Si una ejecución falla, informar el error concreto y la recuperación aplicada. No inventar resultados ni declarar completo un entrenamiento que no terminó. Las preguntas al usuario deben limitarse a información o acceso indispensable; los detalles rutinarios de implementación se resuelven con el criterio acordado.

## 16. Documentación oficial de referencia

Consultar las APIs correspondientes a las versiones finalmente instaladas:

- [CatBoost: variables categóricas](https://catboost.ai/docs/en/features/categorical-features)
- [LightGBM: API de Python](https://lightgbm.readthedocs.io/en/stable/Python-API.html)
- [scikit-learn: validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Optuna: algoritmos de optimización](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [uv: trabajo con proyectos](https://docs.astral.sh/uv/guides/projects/)

## 17. Instrucción de arranque

> Lee este archivo completo y las instrucciones existentes del repositorio. Inspecciona el proyecto y los cuatro CSV locales. Confirma brevemente el objetivo, el estado real y el primer bloque de trabajo. Después implementa las fases en orden, con los experimentos independientes en paralelo cuando el entorno y el equipo lo permitan. Conserva la validación temporal, la reserva de noviembre para la evaluación final y la prioridad del CSV. Actualiza ESTADO.md con resultados verificables para poder continuar en otra sesión. Avanza hasta generar y verificar outputs/submission.csv, resolviendo los detalles rutinarios sin pedir aprobaciones entre fases y explicando cualquier bloqueo real.
