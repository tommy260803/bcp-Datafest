# Selección congelada antes de noviembre

Fecha Lima: 2026-10-03T13:31:28.770015-05:00. Configuración: configs/final.json; SHA-256: 8b797fd55428d35bbed4ec05f56030ea5ca3315eb286fe7456c9a411fa59cc62.

Componentes elegidos: [{'spec': {'name': 'catboost_trial_006', 'family': 'catboost', 'blocks': [], 'params': {'iterations': 600, 'depth': 3, 'learning_rate': 0.04133944642428183, 'l2_leaf_reg': 31.62562678271973, 'random_strength': 2.3283669032459575, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.2546406282501201}, 'seed': 42}, 'weight': 0.75}, {'spec': {'name': 'features_observed_time', 'family': 'hgb', 'blocks': ['observed_time'], 'params': {}, 'seed': 42}, 'weight': 0.25}]. Media mensual Gini: 0.260545.

Clasificación de candidatos:

                  name   family        blocks  mean_gini  worst_gini    seconds  gini_202608  gini_202609  gini_202610
    catboost_trial_006 catboost      original   0.258349    0.247553  69.167585     0.261989     0.247553     0.265506
    catboost_trial_004 catboost      original   0.257982    0.248340  94.125578     0.261729     0.248340     0.263877
features_observed_time      hgb observed_time   0.257716    0.245825  10.219414     0.258006     0.245825     0.269319
    lightgbm_trial_004 lightgbm observed_time   0.257562    0.245845  12.945129     0.263123     0.245845     0.263719
    catboost_promising catboost observed_time   0.257416    0.243830  81.185813     0.262626     0.243830     0.265791
    lightgbm_trial_005 lightgbm observed_time   0.256957    0.246905  12.710059     0.261624     0.246905     0.262344
     lightgbm_original lightgbm      original   0.256912    0.248645   3.069498     0.257143     0.248645     0.264948
    catboost_trial_002 catboost observed_time   0.256795    0.242726  88.289911     0.265143     0.242726     0.262517
          baseline_hgb      hgb      original   0.256795    0.245847   1.879285     0.259736     0.245847     0.264803
    catboost_trial_000 catboost observed_time   0.256461    0.246721 128.325850     0.260931     0.246721     0.261731
     features_activity      hgb      activity   0.256210    0.238315   1.949643     0.265236     0.238315     0.265079
    lightgbm_trial_003 lightgbm observed_time   0.256203    0.242030  13.200943     0.261637     0.242030     0.264942
     catboost_original catboost      original   0.255893    0.241423  71.562910     0.266708     0.241423     0.259548
    lightgbm_promising lightgbm observed_time   0.255325    0.249302  13.257078     0.256150     0.249302     0.260524
    catboost_trial_005 catboost      original   0.254874    0.244603  93.735637     0.258963     0.244603     0.261057
    lightgbm_trial_002 lightgbm      original   0.254624    0.246176   3.929305     0.258042     0.246176     0.259653
     features_economic      hgb      economic   0.254226    0.243614   1.900683     0.255409     0.243614     0.263654
    catboost_trial_003 catboost observed_time   0.254039    0.237757  87.416286     0.262232     0.237757     0.262128
    catboost_trial_007 catboost      original   0.251738    0.238108 120.006555     0.259819     0.238108     0.257287
      features_linkage      hgb       linkage   0.251604    0.238270   2.051090     0.261734     0.238270     0.254808
    catboost_trial_001 catboost observed_time   0.250700    0.236453  77.359656     0.256164     0.236453     0.259484
      features_history      hgb       history   0.250270    0.239582  10.319044     0.250867     0.239582     0.260359
    lightgbm_trial_000 lightgbm observed_time   0.248804    0.238862  16.725927     0.250784     0.238862     0.256768
    lightgbm_trial_001 lightgbm      original   0.248716    0.236083   6.565541     0.257093     0.236083     0.252972
    lightgbm_trial_007 lightgbm observed_time   0.248195    0.235550  15.295487     0.252650     0.235550     0.256385
    lightgbm_trial_006 lightgbm observed_time   0.244684    0.232490  14.685342     0.254068     0.232490     0.247494
     baseline_logistic logistic      original   0.240609    0.234527   0.868703     0.247852     0.234527     0.239447
       baseline_spline   spline      original   0.239402    0.229074   1.534548     0.243773     0.229074     0.245360

Combinaciones probadas: [{'first': 'catboost_trial_006', 'second': 'features_observed_time', 'first_weight': 0.25, 'mean_gini': 0.2591793759012557}, {'first': 'catboost_trial_006', 'second': 'features_observed_time', 'first_weight': 0.5, 'mean_gini': 0.26049452650684185}, {'first': 'catboost_trial_006', 'second': 'features_observed_time', 'first_weight': 0.75, 'mean_gini': 0.2605452442565364}].

Bootstrap pareado por cliente de mejor combinación frente a mejor individual: {'replicates': 500, 'unit': 'customer (jointly across months)', 'mean_delta': 0.002195778038196594, 'ci95': [0.00022971202274579694, 0.004183681745584231]}. Aceptación predefinida: mean gain >= 0.002 and paired customer bootstrap 95% lower bound > 0; otherwise retain best individual. Ensamble aceptado: True.

Mejor individual frente al segundo: {'replicates': 500, 'unit': 'customer (jointly across months)', 'mean_delta': 0.00036754178190351317, 'ci95': [-0.001763110332975807, 0.0025580284390202567]}. Intervalos exploratorios posteriores a búsqueda: no corrigen el optimismo por comparar múltiples candidatos y no garantizan superioridad fuera de desarrollo.

Composición diciembre sin historial: 18.575758%; diagnóstico ponderado y por grupos en selection.json. Decisión usa exclusivamente agosto/septiembre/octubre. Iteraciones fijas, sin calibración ni balanceo.
