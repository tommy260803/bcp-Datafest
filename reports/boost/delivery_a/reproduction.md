# Entrega de diciembre regenerada — Candidato A

Archivo local: `outputs/submission.csv` (9.900 filas, UTF-8, LF, columnas
`id_cliente,prediccion`). SHA-256 del archivo **real**:
`9d4c19adc383b78459339133ea0b95118276e36eca6b282eae91273774fed07a`.
Probabilidades entre 0.0309183115514965 y 0.399970886929378; 9.900 valores
distintos. IDs en orden exacto de `data/test.csv`. Los detalles están en
`submission_validation.json` y `prediction_provenance.json`.

Se entrenó la configuración **congelada de A**: 75 % CatBoost original (600
iteraciones), 25 % HGB con `observed_time` (150 iteraciones). Entrenamiento
enero–noviembre de 2026, 110.100 filas. Diciembre no tiene etiquetas ni Gini
calculable localmente. Los cuatro hashes físicos de datos coinciden con los
registrados en A. La evaluación de noviembre no se repitió.

La entrega histórica documentada en Windows tiene SHA-256
`72bb98012af2a91c78569e72ea0ef79adb4e0dbf8dfcc7f865d0f2dad096b4f4`.
**No es byte a byte idéntica** a esta regeneración Linux. El archivo histórico
original no está disponible en esta copia: los extremos de probabilidad coinciden,
pero no es posible comparar todas las predicciones fila a fila. Convertir los
saltos LF del nuevo CSV a CRLF tampoco reproduce ese hash histórico. Pueden
influir diferencias de serialización o pequeñas variaciones numéricas entre
plataformas; no atribuimos la discrepancia exclusivamente al fin de línea.

Una segunda ejecución leyó los modelos guardados y produjo **exactamente el mismo
hash local**, sin reentrenar ni sobrescribir el CSV. El CSV original ya existente
se protege: si otra predicción intenta cambiarlo, el comando exige elegir otro
`--output`. Reportes y modelos de esta regeneración están aislados bajo
`reports/boost/delivery_a/` y `artifacts/boost/delivery_a/`. No se reescribieron
los reportes de A ni se usaron candidatos experimentales de B.
