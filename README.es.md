[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Credit Risk Scoring Lab

Dos proyectos de riesgo crediticio, ambos deliberadamente políglotas pero por razones distintas — uno divide las etapas de un mismo scorecard entre lenguajes por rendimiento y ajuste regulatorio, el otro demuestra interoperabilidad bidireccional genuina entre R y Python. Cada carpeta es autocontenida, con su propio README, dependencias y tests. Este repo reemplaza dos repos separados que antes vivían en este perfil.

## Técnicas

| # | Técnica | Carpeta | Qué hace |
|---|---|---|---|
| 01 | Scorecard políglota (R + Python + C) | [`01-polyglot-scorecard-r-python-c`](01-polyglot-scorecard-r-python-c) | R hace el scorecard regulatorio WOE/IV + regresión logística, Python hace los challengers de ML (XGBoost/LightGBM) + SHAP, C implementa un hot-path de scoring compilado, verificado bit a bit contra el score de R. |
| 02 | Interoperabilidad bidireccional R↔Python | [`02-bidirectional-r-python-interop`](02-bidirectional-r-python-interop) | Python se encarga de limpieza de datos y scoring crediticio; R se encarga de análisis de mercado en velas, volatilidad GARCH y calibración empírica de LGD (Tobit/GAM); conectados en ambas direcciones vía `reticulate` (R llama a Python) y `rpy2` (Python llama a R), con stress test contra datos macro reales de Chile bajo un enfoque IFRS9/Basilea III. |

## Por qué un repo en vez de dos

Ambos proyectos son reales, ejecutables y probados de forma independiente — esto no es esconder alcance, es representarlo con precisión. Dos repos titulados en torno a "riesgo crediticio" y "R+Python" se leen como duplicación; un laboratorio hace visible la distinción real: un proyecto trata sobre *dónde* vive cada etapa de un pipeline según el lenguaje (una decisión de diseño de sistemas), el otro trata sobre *cómo* dos lenguajes se llaman directamente entre sí (una decisión de ingeniería de interoperabilidad).

## Cómo correr una técnica

Cada carpeta es autocontenida — ver su propio README para el setup exacto y el entry point, resultados reales de una corrida real, y cualquier hallazgo negativo honesto.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
