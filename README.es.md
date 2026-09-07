[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Credit Risk Scoring Lab

Cinco enfoques autocontenidos para la misma pregunta — *.que tan probable es que este deudor caiga en default, y que se hace con eso?* — cada uno respondiendola con un metodo distinto, y cada uno reportando lo que su metodo cuesta ademas de lo que aporta. Cada carpeta tiene su propio README, dependencias, tests y un pipeline que corre de punta a punta con un solo comando.

## Tecnicas

| # | Tecnica | Carpeta | Que hace |
|---|---|---|---|
| 01 | Scorecard poliglota (R + Python + C) | [`01-polyglot-scorecard-r-python-c`](01-polyglot-scorecard-r-python-c) | R hace el scorecard regulatorio WOE/IV + regresion logistica, Python hace los challengers de ML (XGBoost/LightGBM) + SHAP, C implementa un hot-path de scoring compilado, verificado bit a bit contra el score de R. |
| 02 | Interoperabilidad bidireccional R↔Python | [`02-bidirectional-r-python-interop`](02-bidirectional-r-python-interop) | Python se encarga de limpieza de datos y scoring crediticio; R se encarga de analisis de mercado en velas, volatilidad GARCH y calibracion empirica de LGD (Tobit/GAM); conectados en ambas direcciones via `reticulate` (R llama a Python) y `rpy2` (Python llama a R), con stress test contra datos macro reales de Chile bajo un enfoque IFRS9/Basilea III. |
| 03 | PD de por vida con analisis de supervivencia | [`03-survival-lifetime-pd-term-structure`](03-survival-lifetime-pd-term-structure) | Modela *cuando* llega el default, no solo si llega: riesgos proporcionales de Cox implementado desde cero (verosimilitudes parciales de Efron y Breslow, gradiente y hessiano analiticos, diagnostico de Schoenfeld) mas un modelo de hazard en tiempo discreto, convertidos en una estructura temporal de PD para IFRS 9 — el 50,5% del riesgo de vida completa llega despues del mes 12. |
| 04 | Scorecard bayesiano jerarquico | [`04-bayesian-hierarchical-partial-pooling`](04-bayesian-hierarchical-partial-pooling) | Pooling parcial sobre 32 segmentos comerciales, muestreado con un Gibbs escrito desde cero sobre aumentacion Polya-Gamma. Cada PD es una distribucion posterior; el error de los efectos de segmento baja 39% contra no hacer pooling y contra el pooling completo, y la politica de aprobacion puede usar esa incertidumbre. |
| 05 | Restricciones monotonas + decision conforme | [`05-monotonic-constraints-conformal-decisioning`](05-monotonic-constraints-conformal-decisioning) | Un modelo que un supervisor puede aceptar y que sabe cuando abstenerse: gradient boosting con restricciones de monotonia, auditado por perturbacion contrafactual (97% de los solicitantes tenia una violacion antes de restringir, 0% despues, sin costo de precision), envuelto en un predictor conforme Mondrian que convierte la PD en aprobar / revision manual / rechazar con una garantia de error sin supuestos distribucionales. |

## Que las une

Cada tecnica ataca un modo de falla distinto de un modelo de PD a secas:

- **01** — el mismo scorecard tiene que ser interpretable *y* rapido en produccion, y esas dos cosas tiran para lados distintos.
- **02** — el riesgo de credito no se estima aislado del riesgo de mercado, y las herramientas de cada uno viven en lenguajes distintos.
- **03** — una PD a 12 meses no dice nada sobre *cuando* llega el riesgo, que es justo de lo que depende la provision bajo IFRS 9.
- **04** — un solo modelo para una cartera heterogenea esta mal, y un modelo por segmento es ruido; y una estimacion puntual esconde cuanto sabe realmente el modelo.
- **05** — un modelo que contradice al dominio no se aprueba, y un modelo que no puede abstenerse automatiza justo las decisiones que no deberia estar tomando.

## Como correr una tecnica

Cada carpeta es autocontenida — ver su propio README para el setup exacto y el entry point, resultados reales de una corrida real, y sus hallazgos negativos honestos.

```bash
cd 03-survival-lifetime-pd-term-structure    # o cualquier otra
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
pytest -q
```

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Codigo: MIT — ver [LICENSE](LICENSE)
