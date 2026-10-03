# Evaluación final noviembre

Configuración congelada 8b797fd55428d35bbed4ec05f56030ea5ca3315eb286fe7456c9a411fa59cc62 desde 2026-10-03T13:31:28.770015-05:00. Train enero–octubre; evaluación noviembre. Variables, pesos, semillas e iteraciones fijados en desarrollo.

Gini global 0.233079; AUC 0.616540; media desarrollo 0.260545.

- rows: 9500
- positives: 1439
- gini: 0.233079181939587
- auc: 0.6165395909697935
- no_history: {'rows': 728, 'positives': 143, 'gini': 0.19146494531109903}
- history: {'rows': 8772, 'positives': 1296, 'gini': 0.23310994358903248}
- composition_weighted_gini: 0.2319544623873684
- top20_count: 1900
- top20_positives: 428
- top20_recall: 0.2974287699791522

Una evaluación de un solo mes; no es garantía del Gini de diciembre. Noviembre tiene menor proporción sin historial que diciembre; el AUC global reponderado es solo diagnóstico de composición. No se ajustan hiperparámetros tras consultar este resultado.
