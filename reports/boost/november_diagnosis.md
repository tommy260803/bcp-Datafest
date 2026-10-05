# Diagnóstico descriptivo de noviembre — Candidato A

Se usan únicamente predicciones guardadas. No se entrena ni evalúa B. Diciembre solo aporta composición y distribuciones observables, sin etiquetas ni scores.

```text
 month  rows  no_history_fraction  positive_rate     gini  gini_history  gini_no_history
202608 10050             0.175124       0.144179 0.264211      0.259611         0.258215
202609  9900             0.131212       0.140707 0.249065      0.248302         0.231895
202610 10400             0.182019       0.156538 0.268360      0.258705         0.294003
202611  9500             0.076632       0.151474 0.233079      0.233110         0.191465
202612  9900             0.185758            NaN      NaN           NaN              NaN
```

IC95% exploratorio de noviembre menos media desarrollo, agrupando por cliente: [-0.06248087603088249, 0.007150032648140232].

Gini noviembre 0.233079; reponderado por composición diciembre 0.231954. La ponderación por historial no explica por sí sola la caída. El grupo con historial también empeora. Las tasas, segmentos, cuantiles y frecuencias completas están en drift_*.csv. Gini por segmento se reporta solo con al menos 30 ejemplos de cada clase; la restricción del rango de riesgo puede disminuir el Gini intragrupo. Este diagnóstico no distingue causalmente drift, selección previa y variación aleatoria.
