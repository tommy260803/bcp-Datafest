# Ronda 2 — HGB acotado

Ocho configuraciones predeclaradas; hasta dos reemplazos con pesos de A intactos. Plan lógico: 7a30652db4ac4c80e7f975b7a556d47b0b9c683bda1395c93efc015bbf55adcb.

```text
                             name representation  mean_gini  worst_gini  gain_vs_same_features  shortlist_gate_passed  gini_202608  gini_202609  gini_202610
          r2_hgb_observed_shallow  observed_time   0.261742    0.252397               0.004026                   True     0.264583     0.252397     0.268248
          r2_hgb_original_shallow       original   0.259734    0.253887               0.002939                   True     0.261345     0.253887     0.263969
              r2_hgb_lags_shallow  temporal_lags   0.258495    0.252184               0.001156                   True     0.256875     0.252184     0.266427
      r2_hgb_observed_regularized  observed_time   0.255490    0.238951              -0.002227                  False     0.259968     0.238951     0.267550
          r2_hgb_lags_regularized  temporal_lags   0.254646    0.244747              -0.002693                  False     0.255828     0.244747     0.263362
             r2_hgb_observed_wide  observed_time   0.254549    0.246965              -0.003167                  False     0.255814     0.246965     0.260868
r2_hgb_lags_strong_regularization  temporal_lags   0.253550    0.237898              -0.003789                  False     0.256993     0.237898     0.265758
             r2_hgb_original_wide       original   0.249009    0.238197              -0.007786                  False     0.250846     0.238197     0.257983
```

A: 0.260545; diferencia máxima frente a predicciones históricas: 9.99e-16.

Shortlist: ['r2_hgb_observed_shallow', 'r2_hgb_original_shallow'].

- r2_hgb_observed_shallow: media 0.260718; ganancia A +0.000173; peor mes 0.249786; gain below the predeclared minimum.
- r2_hgb_original_shallow: media 0.260368; ganancia A -0.000177; peor mes 0.249753; gain below the predeclared minimum.

**Decisión: retain A; stop this HGB round.** Noviembre no evaluado para B. Los intervalos, si se calculan, son exploratorios sin corrección por selección múltiple. Leaderboard y métricas por historial completos en los archivos de esta ronda.
