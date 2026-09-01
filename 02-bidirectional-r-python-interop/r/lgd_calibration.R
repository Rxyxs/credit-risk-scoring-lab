# Calibracion empirica de LGD (Loss Given Default) sobre el ciclo macroeconomico
# REAL de Chile (data/chile_macro_indicators.csv, Banco Mundial -- ver
# python/fetch_macro_data.py). LGD es una fraccion en [0,1] con masa puntual real
# en ambos extremos (recuperacion total -> LGD=0; garantia inexistente o perdida
# total -> LGD=1), exactamente el caso de uso clasico de una regresion Tobit
# censurada en dos colas (no OLS, que ignoraria la censura y sesgaria los
# coeficientes -- ni una regresion logistica, que no modela el nivel continuo).
#
# El panel de prestamos en default es sintetico (no existe un dataset publico de
# recuperaciones de creditos chilenos), pero la recuperacion de cada prestamo
# depende causalmente del año de default a traves de las variables macro REALES
# (crecimiento del PIB, desempleo) -- mismo patron que el resto del portafolio:
# estructura causal conocida e inyectada para que la calibracion tenga señal real
# que recuperar, verificable contra el supuesto de partida.

library(AER)      # tobit()
library(mgcv)     # gam(), termino suave no lineal
library(dplyr)

set.seed(42)

simulate_defaulted_loan_panel <- function(macro_df, loans_per_year = 80) {
  garantia_tipos <- c("Hipotecaria", "Prendaria", "Sin Garantia")
  garantia_probs <- c(0.30, 0.25, 0.45)

  panel <- macro_df %>%
    rowwise() %>%
    do({
      anio_actual <- .$anio
      n <- loans_per_year
      tipo_garantia <- sample(garantia_tipos, n, replace = TRUE, prob = garantia_probs)

      # % de la exposicion cubierta por garantia -- 0 si no hay garantia, alto y
      # variable si es hipotecaria (bien colateralizada), moderado si es prendaria.
      colateral_pct <- case_when(
        tipo_garantia == "Hipotecaria" ~ pmin(100, rnorm(n, 85, 15)),
        tipo_garantia == "Prendaria" ~ pmin(100, pmax(0, rnorm(n, 45, 20))),
        TRUE ~ 0
      )
      colateral_pct <- pmax(0, colateral_pct)

      # Recuperacion latente: colateral domina, pero el CICLO MACRO real del año de
      # default mueve el valor de liquidacion de la garantia y la capacidad de pago
      # post-default (recesion = colateral se liquida mas barato y mas lento;
      # desempleo alto = menor probabilidad de renegociacion/pago parcial).
      ruido <- rnorm(n, 0, 12)
      recuperacion_pct <- colateral_pct * 0.75 +
        2.0 * .$pib_crecimiento_pct -
        1.5 * .$desempleo_pct +
        15 + ruido
      recuperacion_pct <- pmin(100, pmax(0, recuperacion_pct))

      data.frame(
        anio_default = anio_actual,
        tipo_garantia = tipo_garantia,
        colateral_pct = colateral_pct,
        pib_crecimiento_pct = .$pib_crecimiento_pct,
        desempleo_pct = .$desempleo_pct,
        inflacion_pct = .$inflacion_pct,
        lgd = 1 - recuperacion_pct / 100
      )
    }) %>%
    ungroup()

  panel$tipo_garantia <- factor(panel$tipo_garantia, levels = garantia_tipos)
  panel
}

#' Tobit censurado en [0,1] -- el modelo de calibracion empirica solicitado.
fit_lgd_tobit <- function(panel) {
  AER::tobit(
    lgd ~ colateral_pct + tipo_garantia + pib_crecimiento_pct + desempleo_pct,
    left = 0, right = 1, data = panel
  )
}

#' GAM con termino suave sobre el ciclo del PIB -- capta no linealidad (ej. el
#' deterioro de LGD puede acelerarse desproporcionadamente en recesiones severas,
#' algo que el indice lineal del Tobit no puede representar por construccion).
fit_lgd_gam <- function(panel) {
  mgcv::gam(
    lgd ~ s(pib_crecimiento_pct, k = 6) + s(desempleo_pct, k = 6) +
      colateral_pct + tipo_garantia,
    data = panel, family = mgcv::betar(link = "logit")
  )
}

if (sys.nframe() == 0) {
  macro <- readr::read_csv("data/chile_macro_indicators.csv", show_col_types = FALSE)

  cat(sprintf("=== 1/3 Simulando panel de prestamos en default (%d anios reales x ~80 prestamos/anio) ===\n", nrow(macro)))
  panel <- simulate_defaulted_loan_panel(macro)
  cat(sprintf("  %d prestamos en default simulados, LGD medio bruto: %.1f%%\n", nrow(panel), 100 * mean(panel$lgd)))

  cat("\n=== 2/3 Calibrando LGD empirica: Tobit censurado [0,1] vs. GAM no lineal ===\n")
  tobit_fit <- fit_lgd_tobit(panel)
  cat("\n--- Tobit (AER::tobit) ---\n")
  print(summary(tobit_fit))

  gam_fit <- fit_lgd_gam(panel)
  cat("\n--- GAM (mgcv::gam, familia Beta) ---\n")
  print(summary(gam_fit))

  # Chequeo de signo economico: crecimiento del PIB debe REDUCIR LGD (mejor
  # liquidacion de garantias), desempleo debe AUMENTARLA -- si el Tobit no
  # recupera esto con el signo correcto, la calibracion no es utilizable para
  # estres macro y no deberia usarse aguas abajo sin investigarlo.
  coefs <- coef(tobit_fit)
  pib_coef <- coefs[["pib_crecimiento_pct"]]
  desempleo_coef <- coefs[["desempleo_pct"]]
  cat(sprintf(
    "\nChequeo de signo economico -- PIB: %.4f (%s), Desempleo: %.4f (%s)\n",
    pib_coef, ifelse(pib_coef < 0, "OK, reduce LGD", "INESPERADO"),
    desempleo_coef, ifelse(desempleo_coef > 0, "OK, aumenta LGD", "INESPERADO")
  ))
  stopifnot(pib_coef < 0, desempleo_coef > 0)

  dir.create("output/models", showWarnings = FALSE, recursive = TRUE)
  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)
  saveRDS(tobit_fit, "output/models/lgd_tobit_fit.rds")
  saveRDS(gam_fit, "output/models/lgd_gam_fit.rds")
  readr::write_csv(panel, "data/defaulted_loan_panel_synthetic.csv")

  cat("\n=== 3/3 LGD promedio predicha por escenario de PIB (colateral/garantia tipicos) ===\n")
  grid <- expand.grid(
    pib_crecimiento_pct = c(-6.0, 0.0, 4.0),
    desempleo_pct = 8.0,
    colateral_pct = 50,
    tipo_garantia = factor("Prendaria", levels = levels(panel$tipo_garantia))
  )
  grid$lgd_tobit <- pmin(1, pmax(0, predict(tobit_fit, newdata = grid)))
  grid$lgd_gam <- as.numeric(predict(gam_fit, newdata = grid, type = "response"))
  print(grid)

  readr::write_csv(grid, "output/tables/lgd_calibration_scenario_check.csv")
  cat("\nModelos guardados en output/models/lgd_tobit_fit.rds y lgd_gam_fit.rds\n")
}
