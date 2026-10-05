# Ronda 3 — CatBoost original acotado

Seis configuraciones predeclaradas; hasta dos combinaciones 75/25. Plan: 84b2978acd979042992b5c633bbe698aaf1870c55f18ae664737b0606a45fb64.

```text
                     name  mean_gini  worst_gini  gain_vs_catboost_a  shortlist_gate_passed  gini_202608  gini_202609  gini_202610
 r3_catboost_depth3_l2_10   0.258420    0.245531            0.000070                  False     0.264452     0.245531     0.265276
r3_catboost_depth3_slower   0.257520    0.245262           -0.000830                  False     0.262901     0.245262     0.264397
 r3_catboost_depth4_l2_60   0.257479    0.242887           -0.000871                  False     0.264582     0.242887     0.264968
 r3_catboost_depth3_l2_60   0.257029    0.245756           -0.001321                  False     0.262051     0.245756     0.263279
       r3_catboost_depth2   0.255902    0.245178           -0.002448                  False     0.260860     0.245178     0.261667
 r3_catboost_depth2_l2_60   0.254355    0.243854           -0.003994                  False     0.260632     0.243854     0.258580
```

A reproducido 0.260545; diferencia máxima 9.99e-16.

Shortlist: None.

Individual: None.



**Decisión: retain A; stop this CatBoost round.** Noviembre no evaluado para B. Los intervalos, cuando se calculan, son exploratorios y no corrigen la selección acumulada de candidatos.
