# Comparación controlada de bloques

Mismo HGB, cortes y presupuesto; se cambia únicamente el bloque. Historial usa shift antes de rolling; meses en índice calendario; orden restaurado. Campos estáticos sin tendencias artificiales.

                  name family        blocks  mean_gini  worst_gini   seconds  gini_202608  gini_202609  gini_202610
features_observed_time    hgb observed_time   0.257716    0.245825 10.219414     0.258006     0.245825     0.269319
          baseline_hgb    hgb      original   0.256795    0.245847  1.879285     0.259736     0.245847     0.264803
     features_activity    hgb      activity   0.256210    0.238315  1.949643     0.265236     0.238315     0.265079
     features_economic    hgb      economic   0.254226    0.243614  1.900683     0.255409     0.243614     0.263654
      features_linkage    hgb       linkage   0.251604    0.238270  2.051090     0.261734     0.238270     0.254808
      features_history    hgb       history   0.250270    0.239582 10.319044     0.250867     0.239582     0.260359

Bloques prometedores provisionales: ['observed_time']. Cambios pequeños requieren análisis pareado antes de interpretar mejoras como estables.
