# Candidato B — desarrollo

Selección exclusivamente agosto–octubre. Sin tuning inicial. Una comparación CatBoost exploratoria puede ejecutarse aunque falle la puerta HGB; LightGBM solo se abre si CatBoost mejora de forma consistente. Se comprueba una sola sustitución del HGB de A por el mejor HGB dinámico, conservando los pesos de A.

```text
                               name   family             blocks  mean_gini  worst_gini   seconds                                                      fingerprint  gini_202608  gini_202609  gini_202610
    a_reproduced_catboost_trial_006 catboost           original   0.258349    0.247553 89.187319 fa323385f69b69a50705245ca35d0cfb1b469e1a9c587f836a564b517d7e2fc8     0.261989     0.247553     0.265506
                b_hgb_observed_time      hgb      observed_time   0.257716    0.245825 44.021130 efde98e1692a6b0a1c5ef7a90d022c30180361b5517b6383af5f3f281ae2172e     0.258006     0.245825     0.269319
a_reproduced_features_observed_time      hgb      observed_time   0.257716    0.245825 48.392297 eef46c8dfa7542b4f9caa7bbc078ab48ff2f21d1d018d23a18c3e4a34033b9af     0.258006     0.245825     0.269319
           b_hgb_temporal_structure      hgb temporal_structure   0.257716    0.245825  6.571537 1e5dd6ec28262b43e1bc068cfd0af48532e2303fa3b595dc2b31aada6d047080     0.258006     0.245825     0.269319
                b_hgb_temporal_lags      hgb      temporal_lags   0.257339    0.251937  7.586172 8b7330cf5d59decff92f8a3ef8927dd236fbd5dd78596f437550b9c2f2d4c1e6     0.251962     0.251937     0.268117
                     b_hgb_original      hgb           original   0.256795    0.245847  5.961089 0149d09425d04da824af670dceebb59c8bf4738b34e08fc9065e5cbd6385e3b1     0.259736     0.245847     0.264803
                  b_hgb_temporal_v2      hgb        temporal_v2   0.256387    0.246690  8.632393 bd37d1e3a51920cfaafd8e94493efb78f844051bb9c887101871aa62f8a220ac     0.257057     0.246690     0.265415
           b_catboost_temporal_lags catboost      temporal_lags   0.255367    0.243923 89.560187 4d9d251c2f82ae007c8a1afcd30696d1f3a1eb7683de889ee421c7a744d998aa     0.259021     0.243923     0.263159
```

A reproducido: 0.260545. Mejor bloque HGB: temporal_structure; puerta: False.

B: 0.257716; peor mes 0.245825. Bootstrap pareado B−A: {'replicates': 500, 'unit': 'customer (jointly across months)', 'valid_replicates': 500, 'invalid_replicates': 0, 'mean_delta': -0.0028287545275911263, 'ci95': [-0.009889241638790824, 0.0041996966282515594]}.

Decisión: **retain A; no November evaluation for B**. 

Combinación fija: [{'strategy': 'fixed A weights; replace HGB with best dynamic HGB', 'metrics': {'mean_gini': 0.2612390395955622, 'worst_gini': 0.25092709006754377, 'folds': [{'rows': 10050, 'positives': 1449, 'gini': 0.26327615780308333, 'auc': 0.6316380789015417, 'no_history': {'rows': 1760, 'positives': 309, 'gini': 0.25898219953207113}, 'history': {'rows': 8290, 'positives': 1140, 'gini': 0.25824512329775495}, 'composition_weighted_gini': 0.26350264915492416, 'top20_count': 2010, 'top20_positives': 499, 'top20_recall': 0.3443754313319531, 'month': 202608}, {'rows': 9900, 'positives': 1393, 'gini': 0.25092709006754377, 'auc': 0.6254635450337719, 'no_history': {'rows': 1299, 'positives': 218, 'gini': 0.2340001188162506}, 'history': {'rows': 8601, 'positives': 1175, 'gini': 0.25023660399630954}, 'composition_weighted_gini': 0.25092706599450265, 'top20_count': 1980, 'top20_positives': 465, 'top20_recall': 0.3338119167264896, 'month': 202609}, {'rows': 10400, 'positives': 1628, 'gini': 0.2695138709160596, 'auc': 0.6347569354580298, 'no_history': {'rows': 1893, 'positives': 341, 'gini': 0.2943094899779304}, 'history': {'rows': 8507, 'positives': 1287, 'gini': 0.2600399907879132}, 'composition_weighted_gini': 0.26969138804739257, 'top20_count': 2080, 'top20_positives': 542, 'top20_recall': 0.33292383292383293, 'month': 202610}]}, 'components': [{'spec': {'name': 'catboost_trial_006', 'family': 'catboost', 'blocks': [], 'params': {'iterations': 600, 'depth': 3, 'learning_rate': 0.04133944642428183, 'l2_leaf_reg': 31.62562678271973, 'random_strength': 2.3283669032459575, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.2546406282501201}, 'seed': 42}, 'weight': 0.75}, {'spec': {'name': 'b_hgb_temporal_lags', 'family': 'hgb', 'blocks': ['temporal_lags'], 'params': {}, 'seed': 42}, 'weight': 0.25}], 'gain_vs_best_individual': 0.0006937953390258045, 'paired_vs_best_individual': None, 'accepted': False}].

Los intervalos son exploratorios y no corrigen selección múltiple. No se evalúa noviembre con ninguna variante de B desde este comando.
