<div align="center">

# 🏦 Las afirmaciones del laboratorio sobre una cartera real

**Las tecnicas 01, 03, 05, 06, 10, 12 y 22 se construyeron y midieron sobre carteras simuladas. Esta carpeta corre su propio codigo sobre 621.022 creditos reales de Lending Club (2007-2015) y reporta que hallazgos sobreviven, cuales se achican y cuales se dan vuelta.**

🌐 **[English](README.md)** | **[Español](README.es.md)**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Datos](https://img.shields.io/badge/datos-Lending%20Club%202007--2015%20%2B%20FRED-2C5F8A)](#los-datos)
[![Tests](https://img.shields.io/badge/tests-22%20pasando-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](../LICENSE)

</div>

---

## Por que existe esta carpeta

Casi todo este laboratorio corre sobre datos simulados por una razon
declarada: sus afirmaciones son sobre *recuperar* algo (un coeficiente, una
correlacion, una brecha que se planto), y eso solo se puede verificar contra
una verdad escrita de antemano. Pero varias afirmaciones no son
recuperaciones. "El scorecard gano en precision", "la restriccion monotona no
cuesta nada", "la mitad de la cartera va a revision manual", "la mitad del
riesgo de vida llega despues del mes 12", "el PSI detecta el cambio": esas se
pueden verificar sobre una cartera real, donde nadie controla el proceso que
genera los datos.

Aca no se reimplementa nada. [`src/lab.py`](src/lab.py) importa el binning
optimo y el scorecard de la tecnica 06, la auditoria de monotonia y el
predictor conforme de la 05, Kaplan-Meier de la 03, los estimadores de
correlacion de la 10 y el PSI de la 12; esta carpeta solo pone los datos y
las comparaciones.

Preparar esta carpeta tambien destapo un error en la tecnica 10: su metodo de
momentos ignoraba el ruido binomial de las cohortes finitas, lo que importa en
una cartera real cuyas cohortes de 2007 tienen unas pocas docenas de
creditos. Se corrigio en la propia tecnica 10 (ver su seccion de
granularidad), y el estimador corregido es el que se usa abajo.

## Los datos

- **Lending Club**, todos los creditos que la plataforma otorgo hasta el
  cuarto trimestre de 2018 (Kaggle `wordsforthewise/lending-club`,
  `accepted_2007_to_2018Q4.csv.gz`). Se quedan los creditos a 36 meses
  otorgados entre 2007 y 2015, que ya vencieron todos en el archivo, asi que
  su resultado se conoce. Quedan **621.022 creditos** (se sacaron 147 que
  seguian abiertos), con **13,95%** castigados.
- **Tasa de desempleo de EE.UU.**, serie `UNRATE` de FRED, para el analisis
  por cosecha.
- **Fuera de tiempo**: los modelos se ajustan con 2007-2013 (175.426
  creditos, 12,6% malos), se revisan con 2014 (162.570, 13,7%) y se prueban
  con 2015 (283.026, 14,9%).
- **Variables**: solo lo que se sabia al aprobar el credito: FICO, ingreso,
  carga financiera, uso y saldo de lineas rotativas, consultas, moras,
  registros publicos, numero de cuentas, antiguedad crediticia, monto del
  credito sobre el ingreso, tenencia de vivienda, proposito, antiguedad
  laboral y verificacion del ingreso. El grado, sub-grado y tasa de interes
  que asigno Lending Club quedan fuera de todo modelo y se usan como
  referencia: son con lo que el prestamista puso precio. No se lee nada
  registrado despues del desembolso, salvo la fecha del ultimo pago, que solo
  usa el analisis del momento del default.

## Que sobrevivio

| Afirmacion del laboratorio | Tecnica | Simulado | Real (Lending Club) | Veredicto |
|---|---|---|---|---|
| El binning optimo recupera mas IV que los deciles | 06 | +13% | **+6,5%** | se mantiene, menor |
| Un arbol greedy le gana a la DP fuera de tiempo | 06 | +0,7 pp de AUC | **+0,03 pp** | se reduce a ruido |
| El scorecard interpretable gana en precision | 01 | 0,733 vs. 0,709-0,716 | **0,647 vs. 0,660** | se da vuelta |
| La restriccion monotona no cuesta nada | 05 | 0,7695 vs. 0,7655 | **0,6602 vs. 0,6595** | se mantiene |
| El boosting sin restriccion rompe la monotonia | 05 | 97,15% de los solicitantes | **90,5%** | se mantiene |
| Volumen de revision conforme con α = 0,10 | 05 | 50,1% | **63,7%** | peor |
| La cobertura conforme se cumple | 05 | ≥ 90% | **90,4%**, calibrado un ano antes | se mantiene |
| Riesgo de vida que llega despues del mes 12 | 03 | 50,5% | **72,4%** | mayor |
| Correlacion de activos de un grado de riesgo | 10 | 0,12-0,24 (fijada en la formula corporativa de Basilea) | **0,00-0,06** | mucho menor |
| El monitoreo de drift detecta lo que importa | 12, 22 | PSI 0,82 en la variable desplazada | **PSI maximo 0,095; el gatillo por AUC se dispara todos los trimestres** | no ve el cambio real |

El patron es el mismo del resto de este portafolio: con datos reales la senal
es mas debil (el Information Value total del scorecard es 0,50, contra 2,12 en
la tecnica 06), asi que las diferencias entre metodos se achican, y los
umbrales y parametros elegidos sobre datos simulados dejan de significar lo
que significaban ahi.

## 1 · Binning y eleccion de modelo (tecnicas 01, 05, 06)

![Comparacion de modelos](outputs/plots/comparacion_modelos.png)

| Modelo, ajustado con 2007-2013 | AUC 2015 | KS 2015 | Brier 2015 | Violaciones de monotonia |
|---|---|---|---|---|
| Scorecard, bins equifrecuentes | 0,6451 | | | 3 variables no monotonas |
| Scorecard, bins DP libres | 0,6466 | | | 2 variables no monotonas |
| **Scorecard, bins DP monotonos** | 0,6467 | 0,211 | 0,1230 | ninguna |
| Scorecard, bins de arbol | 0,6470 | | | 4 variables no monotonas |
| Gradient boosting sin restriccion | 0,6595 | 0,230 | 0,1221 | 90,5% de los solicitantes |
| **Gradient boosting monotono** | **0,6602** | **0,232** | 0,1221 | **0%** |
| Sub-grado de Lending Club (referencia) | *0,6787* | *0,262* | | |

- **Los cuatro metodos de binning quedan a menos de 0,2 puntos de AUC.** La DP
  sigue encontrando mas Information Value que los deciles (0,504 contra
  0,473) y sigue siendo el unico metodo sin variables no monotonas, asi que su
  argumento en la tecnica 06 (una tarjeta que un comite puede defender)
  sobrevive. El argumento de precision no separa a los metodos en esta
  cartera.
- **El scorecard no gana.** El scorecard de la tecnica 01 le gano a todos los
  modelos de machine learning en su cartera simulada; aca el gradient
  boosting le saca 1,3 puntos de AUC, con las mismas variables.
- **La restriccion monotona sigue siendo gratis.** La afirmacion central de la
  tecnica 05 se sostiene con datos reales: el modelo restringido es tan bueno
  como el libre (0,6602 contra 0,6595), y el libre contradice la direccion
  esperada de al menos una variable para el 90,5% de los solicitantes.
- **El grado de Lending Club les gana a todos** (0,679). El prestamista vio el
  informe de credito completo y su propia historia; estos modelos ven 17
  variables resumidas. Es el techo honesto de este ejercicio, no una meta.
- **Todos los modelos subestiman 2015.** PD media de 12,7-13,0% contra 14,9%
  observada: la cartera se volvio mas riesgosa de una forma que las variables
  del solicitante no muestran (seccion 5).

## 2 · Decisiones conformes (tecnica 05)

El predictor conforme Mondrian de la tecnica 05, sobre el modelo monotono, con
α = 0,10:

| | Simulado (05) | Calibrado con 2014, probado con 2015 | Calibrado y probado dentro de 2015 |
|---|---|---|---|
| Cobertura (buenos / malos) | 90% por construccion | 90,6% / 89,4% | 90,0% / 90,1% |
| Aprobado automatico | 32,7% | 25,2% | 24,0% |
| **Revision manual** | 50,1% | **63,7%** | 64,2% |
| Rechazo automatico | — | 11,1% | 11,8% |
| Error en las decisiones automaticas | 19,8% | 26,4% | 27,9% |
| Tasa de malos entre los aprobados | 7,6% | 6,3% | 6,1% |

La garantia se cumple incluso a traves de un ano en que la tasa de default
subio un punto (cobertura de los malos 89,4%, apenas bajo 90%). Lo que cuesta
es volumen: con un AUC de 0,66 el modelo duda de dos tercios de la cartera,
asi que con α = 0,10 dos tercios van a un analista.

## 3 · Cuando ocurren los defaults (tecnica 03)

![Momento del default](outputs/plots/momento_del_default.png)

La fecha del default es el mes del ultimo pago de un credito castigado, el
inicio de la mora (Lending Club lo castiga unos cinco meses despues). El
Kaplan-Meier de la tecnica 03, con los prepagos como censura:

- PD a 12 meses **5,3%**, en los 36 meses de vida **19,1%**: un multiplo de
  **3,6×** (2,0× en la tecnica 03), asi que el **72,4%** del riesgo de vida
  llega despues del mes 12 (50,5% simulado). De los creditos que de verdad
  cayeron, el 64,8% hizo su ultimo pago despues del mes 12; la mediana es el
  mes 16.
- Si cada default se fecha a los 90 dias de mora, el 82,7% del riesgo de vida
  queda despues del mes 12.
- La proporcion baja con el grado: 80% en los creditos A, 52% en los G. Los
  deudores mas seguros caen mas tarde.

Para IFRS 9 esta es la direccion importante: una PD a 12 meses subestima la
perdida de vida de estos creditos mas de lo que sugeria la cartera simulada
del laboratorio.

## 4 · Cosechas y correlacion de activos (tecnica 10)

![Cosechas y correlacion](outputs/plots/cosechas_y_correlacion.png)

Cada grado se sigue trimestre de originacion por trimestre (33-34 trimestres
para los grados A-E, 17 para F; G tiene muy pocos creditos por trimestre y
queda fuera). El metodo de momentos de la tecnica 10 convierte la dispersion
de las tasas de default de cada grado en una correlacion de activos.

| Grado | PD a 36 meses | rho con el tamano de las cohortes | rho sin el | rho sin tendencia lineal | Basilea, otros minoristas | Basilea, corporativo |
|---|---|---|---|---|---|---|
| A | 5,6% | 0,000 | 0,016 | 0,000 | 0,097 | 0,166 |
| B | 12,2% | 0,016 | 0,022 | 0,010 | 0,059 | 0,134 |
| C | 17,1% | 0,011 | 0,017 | 0,011 | 0,046 | 0,126 |
| D | 21,6% | 0,017 | 0,024 | 0,018 | 0,039 | 0,122 |
| E | 24,6% | 0,015 | 0,031 | 0,013 | 0,036 | 0,121 |
| F | 30,5% | 0,060 | 0,078 | 0,040 | 0,032 | 0,120 |

(Las formulas de Basilea se evaluan en la PD anualizada, 1 − (1 − PD₃₆)^(1/3).)

- **Las correlaciones son chicas**: 0,00-0,02 en los grados A-E, bastante
  bajo el 0,03-0,16 que Basilea supone para otras exposiciones minoristas y un
  orden de magnitud bajo el 0,12-0,24 de la formula corporativa que simula la
  tecnica 10.
- **El tamano de las cohortes importa en una cartera real.** Sin el, la
  estimacion se duplica en el grado E, y en el grado A toda la dispersion es
  ruido binomial: dado el tamano de las cohortes, no queda nada sistematico.
- **Casi todo el movimiento es a quien le presto Lending Club, no la
  economia.** La cohorte de 2007Q4 cayo en 29,6% en plena recesion, pero desde
  2009 la tasa trimestral se mueve con la participacion de creditos A y B
  (correlacion −0,83) y levemente *en contra* del desempleo (−0,29). El
  modelo de un factor supone que un grado significa el mismo riesgo todos los
  trimestres; Lending Club redibujo sus grados en estos anos. Sacar una
  tendencia lineal de la serie de cada grado casi no cambia A-E y baja F de
  0,060 a 0,040.

## 5 · Monitoreo de drift (tecnicas 12 y 22)

![Estabilidad trimestral](outputs/plots/estabilidad_trimestral.png)

El scorecard monotono ajustado con 2007-2013 puntua cada trimestre de 2014 y
2015. El PSI de la tecnica 12 contra la poblacion de entrenamiento, y la regla
de reentrenamiento de la tecnica 22 (AUC realizado bajo 0,72 o PSI sobre
0,20):

- **El gatillo por AUC se dispara en los 8 trimestres.** El AUC realizado se
  mueve entre 0,640 y 0,650 todo el periodo, el nivel que tiene este modelo en
  esta cartera. El umbral de 0,72 calza con la cadena simulada, donde el AUC
  realizado del Champion era 0,775 (tecnica 21) y el gatillo se disparo en
  0,64; aca 0,64 es lo que marca un modelo que funciona.
- **El gatillo por PSI no se dispara nunca.** El PSI del puntaje no pasa de
  0,013 y ninguna variable vigilada pasa de 0,095: en el papel, los
  solicitantes de 2015 se parecen a los de 2007-2013.
- **Ninguno ve lo que cambio.** La tasa de default observada subio de 12,9% a
  15,4% mientras la PD media del scorecard se quedo cerca de 13% y bajo: en
  2015Q2 subestima por 2,5 puntos. Es un cambio en el resultado de
  solicitantes que se ven iguales, y un indice de estabilidad de poblacion no
  puede detectarlo por construccion. La tecnica 21 reporta la tasa de default
  predicha junto a la observada justamente por esto; la regla de la tecnica 22
  no la usa, y en esta cartera es la unica senal que se movio.

## Lo que esto no muestra

- **Solo creditos aprobados.** Las solicitudes rechazadas por Lending Club no
  tienen resultado, asi que nada de esto dice como les iria a estos modelos con
  toda la poblacion de solicitantes (la pregunta de la tecnica 09).
- **Un prestamista, un pais, un periodo.** 2007-2015 contiene una recesion, al
  comienzo; las correlaciones por cosecha descansan en ella y en 33-34
  trimestres.
- **Un objetivo de vida completa.** La marca de default cubre los 36 meses;
  las tecnicas simuladas usan casi siempre una marca a 12 meses, asi que los
  AUC son comparables en direccion, no al decimal.
- **Una aproximacion de la fecha del default.** La fecha del ultimo pago marca
  el inicio de la mora; Lending Club no publica la fecha del castigo, y la
  version a 90 dias es una aproximacion de la definicion regulatoria.
- **Una corrida.** El gradient boosting usa la configuracion y la semilla de la
  tecnica 05 sin cambios; la diferencia de 0,0007 de AUC entre el modelo
  libre y el restringido esta dentro del ruido entre corridas, que es justo lo
  que dice esa fila.

## Como correrlo

```bash
# ~375 MB desde Kaggle (requiere cuenta de Kaggle) y una serie de FRED, en data/raw (en el gitignore)
kaggle datasets download wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw
curl -o data/raw/UNRATE.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"

python run_pipeline.py     # unos 4 minutos en la CPU de un notebook
pytest -q                  # 22 tests, sin necesidad de datos
```

Los resultados quedan en `outputs/reports/lending_club_results.json`, con las
tablas trimestrales al lado en CSV.

## Archivos

| Archivo | Que hace |
|---|---|
| [`src/lab.py`](src/lab.py) | Carga las tecnicas 03, 05, 06, 10 y 12 desde sus carpetas, para que la cartera real pase por su codigo |
| [`src/data.py`](src/data.py) | Lee el archivo de Lending Club, deja los creditos a 36 meses resueltos, construye las variables al momento de originar y separa por ano |
| [`src/scoring.py`](src/scoring.py) | Cuatro binnings y sus scorecards (06), boosting con y sin restriccion y la auditoria (05), decisiones conformes (05) |
| [`src/timing.py`](src/timing.py) | Curva de default de Kaplan-Meier y la proporcion del riesgo de vida despues del mes 12 (03) |
| [`src/vintages.py`](src/vintages.py) | Cohortes de grado por trimestre, correlacion de activos con y sin tamano de cohorte, formulas de Basilea minorista y corporativa (10) |
| [`src/stability.py`](src/stability.py) | PSI trimestral, AUC realizado y calibracion contra la regla de reentrenamiento de la tecnica 22 (12, 22) |
| [`src/study.py`](src/study.py) | Corre todo y escribe los resultados |
| [`src/plots.py`](src/plots.py) | Los cuatro graficos |
