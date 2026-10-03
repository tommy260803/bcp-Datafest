# Auditoría local

Todos los controles de contrato aprobados. Datos originales sin modificaciones.

Train (110100, 25); test (9900, 24); 24628 clientes en train; 16567 positivos (15.047230%). Test sin historial: 1839/9900.

| Mes | Filas | Positivos | Tasa | Sin historial |
|---|---:|---:|---:|---:|
| 202601 | 10400 | 1521 | 0.146250 | 10400 |
| 202602 | 9800 | 1543 | 0.157449 | 921 |
| 202603 | 10300 | 1527 | 0.148252 | 2043 |
| 202604 | 9600 | 1473 | 0.153438 | 827 |
| 202605 | 10100 | 1584 | 0.156832 | 1973 |
| 202606 | 10350 | 1600 | 0.154589 | 1834 |
| 202607 | 9700 | 1410 | 0.145361 | 950 |
| 202608 | 10050 | 1449 | 0.144179 | 1760 |
| 202609 | 9900 | 1393 | 0.140707 | 1299 |
| 202610 | 10400 | 1628 | 0.156538 | 1893 |
| 202611 | 9500 | 1439 | 0.151474 | 728 |

No hay faltantes, claves duplicadas, múltiples conversiones ni filas posteriores a conversión, incluyendo test. Variación intracliente verificada en train+test:

edad                               0
ingresos                           0
ratio_deuda_ingresos               0
antiguedad_cuenta_meses            0
numero_productos                   0
saldo_promedio                     0
dias_ultima_transaccion            0
antiguedad_direccion_meses         0
visitas_web_ultimos_90_dias        0
distancia_sucursal_km              0
dia_preferido_pago                 0
dias_ultima_interaccion        16769
ocupacion                          0
region                             0
canal_adquisicion                  0
banda_riesgo                       0
dispositivo_principal              0
tiene_tarjeta_credito              0
activo_movil                       0
es_nuevo_cliente                   0
tiene_prestamo                     0
tiene_seguro                       0

Hashes SHA-256:

- `train.csv`: `a714ebcd8587b2cc0f75ae642aa3e4ca5904eea02e249d969a06e8e7380e0567`
- `test.csv`: `81197145e28d1602a0195d93487dc924ae438d8b0eed411cf56cac13d786126f`
- `sample_submission.csv`: `d9e3caf0d5f5cde6fec0a146b4fc69abf98d703148a115ebf2631a61225f0fc5`
- `metaData.csv`: `020ab793205853f11ea4a7d77aae249b38cf24cb74e956b27352bfbf9dc32c0e`

Recursos: {'logical_cpus': 20, 'physical_cpus': 14, 'ram_bytes': 16786370560, 'available_ram_bytes': 4634025984, 'process_rss_bytes': 142454784}.

Tipos, categorías y conteos completos: `audit.json`.
