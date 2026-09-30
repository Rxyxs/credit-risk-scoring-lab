[ 🇺🇸 [Read in English](README.md) ] | [ 🇨🇱 Español ]

# 12 — Monitoreo de Drift (PSI + KS)

Un modelo se degrada de dos formas: la población que le llega cambia (data drift), o la
relación entre las features y el resultado cambia (concept drift). Ninguna de las dos aparece
como una excepción -- las dos se ven como nada, hasta el backtest que la agarra tres meses
tarde. Esta técnica le da al resto del laboratorio una forma de chequear la primera, temprano,
sobre features crudas, sin necesitar un scorecard ya ajustado.

## Qué mide

- **PSI (Population Stability Index)** -- discretiza `expected` (train/base) por sus propios
  cuantiles, y compara qué fracción de `actual` (OOT/scoring) cae en cada bucket:

  `PSI = suma_bucket (pct_actual - pct_expected) * ln(pct_actual / pct_expected)`

  Bandas estándar de la industria: **< 0,10** estable, **0,10–0,25** cambio moderado (alerta),
  **> 0,25** drift severo (recalibración).
- **KS de dos muestras** (`scipy.stats.ks_2samp`) -- compara las CDFs empíricas directamente,
  sin discretizar. Una segunda señal independiente: PSI por deciles puede diluir un corrimiento
  concentrado en un extremo; KS no discretiza, así que también lo agarra.

A diferencia de [`06-optimal-binning-scorecard/src/monitoring.py`](../06-optimal-binning-scorecard/src/monitoring.py)
(PSI/CSI sobre el *score* y los *bins* de un scorecard ya ajustado), este módulo no necesita un
modelo entrenado -- corre sobre cualquier columna numérica, que es el punto: agarrar el drift en
las entradas antes de que le llegue a un modelo.

## API

```python
from src.monitoring.drift import calculate_psi, calculate_ks_drift, generate_drift_report

calculate_psi(expected, actual, num_buckets=10)          # -> float
calculate_ks_drift(expected, actual)                       # -> {"statistic", "p_value", "drift_detected"}
generate_drift_report(baseline_df, scoring_df, features)   # -> dict, una entrada por feature + resumen
```

## Corrida real

```bash
python run_pipeline.py
```

Base sintética (5.000 filas) contra un set de scoring (2.000 filas) con una feature corrida a
propósito -- un movimiento de +1σ en el ingreso mensual medio, las otras dos sin tocar:

| Feature | PSI | Estado | p-valor KS | Drift KS |
|---|---:|:---:|---:|:---:|
| `ingreso_mensual` (corrida +1σ) | **0,873** | 🔴 rojo | 1,9e-194 | True |
| `dti` (sin tocar) | 0,006 | 🟢 verde | 0,454 | False |
| `antiguedad_laboral_meses` (sin tocar) | 0,005 | 🟢 verde | 0,738 | False |

Los dos métodos coinciden en la feature corrida y los dos se quedan callados en las estables --
el punto de correr dos pruebas independientes en vez de una sola.

## Limitación conocida

PSI discretiza por los cuantiles de `expected` mismo. Si `expected` es constante (o casi), sus
cuantiles colapsan en un único corte, y PSI no puede distinguir dos valores distintos en
`actual` entre sí -- reporta 0,0, no un error. No es un bug introducido acá; es inherente a
discretizar por cuantiles de la propia base, y es la misma limitación que tiene el `psi_score`
de `06-optimal-binning-scorecard`. `tests/test_drift_monitoring.py` la documenta como un test
explícito en vez de esconderla.

## Tests

```bash
pytest -v
```

**17 tests**: distribuciones idénticas → PSI≈0 y p-valor KS≈1; muestras independientes de la
misma distribución se quedan bien por debajo del umbral de alerta; un corrimiento severo cruza
0,25 tanto en PSI como en KS; un corrimiento moderado cae específicamente en la banda amarilla
(no solo verde/rojo); los límites de clasificación; y manejo seguro de columnas todas en cero,
NaNs, menos filas que buckets pedidos, y una base constante.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
