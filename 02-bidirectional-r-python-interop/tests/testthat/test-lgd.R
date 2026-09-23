# Tests de la calibracion de LGD (r/lgd_calibration.R). Datos macro
# sinteticos generados on the fly -- no leen data/chile_macro_indicators.csv
# ni escriben nada en output/.

source("../../r/lgd_calibration.R")

macro_sintetico <- function() {
  data.frame(
    anio = 2010:2021,
    pib_crecimiento_pct = c(3.0, -1.0, 2.5, 4.0, -6.0, 1.0, 5.5, -3.0, 0.5, 6.0, -2.0, 2.0),
    desempleo_pct = c(7.0, 9.5, 6.8, 5.5, 13.0, 8.0, 5.0, 11.0, 8.5, 4.5, 10.5, 7.5),
    inflacion_pct = rep(3.0, 12)
  )
}

# --------------------------------------------- simulate_defaulted_loan_panel

test_that("simulate_defaulted_loan_panel produce LGD siempre acotado en [0,1]", {
  set.seed(1)
  panel <- simulate_defaulted_loan_panel(macro_sintetico(), loans_per_year = 40)

  expect_false(anyNA(panel$lgd))
  expect_true(all(panel$lgd >= 0))
  expect_true(all(panel$lgd <= 1))
})

test_that("un ciclo macro con recesiones y booms extremos produce recuperaciones cero y totales", {
  # Con pendientes tan fuertes hacia el PIB/desempleo, el panel sintetico
  # tiene que tocar los bordes de recuperacion (LGD=0) y perdida total
  # (LGD=1) via el pmin/pmax(0,100) de recuperacion_pct -- son observaciones
  # reales del generador, no un caso artificial armado a mano.
  macro_extremo <- data.frame(
    anio = 2000:2005,
    pib_crecimiento_pct = c(15, 15, 15, -20, -20, -20),
    desempleo_pct = c(2, 2, 2, 25, 25, 25),
    inflacion_pct = rep(3, 6)
  )
  set.seed(2)
  panel <- simulate_defaulted_loan_panel(macro_extremo, loans_per_year = 60)

  expect_true(any(panel$lgd == 0))
  expect_true(any(panel$lgd == 1))
  expect_true(all(panel$lgd >= 0 & panel$lgd <= 1))
})

test_that("simulate_defaulted_loan_panel es reproducible con la misma semilla", {
  set.seed(5)
  a <- simulate_defaulted_loan_panel(macro_sintetico(), loans_per_year = 20)
  set.seed(5)
  b <- simulate_defaulted_loan_panel(macro_sintetico(), loans_per_year = 20)
  expect_equal(a, b)
})

# --------------------------------------------------------- fit_lgd_gam

test_that("fit_lgd_gam no falla con observaciones de recuperacion cero o total", {
  set.seed(3)
  panel <- simulate_defaulted_loan_panel(macro_sintetico(), loans_per_year = 40)
  panel$lgd[1] <- 0  # recuperacion total forzada
  panel$lgd[2] <- 1  # recuperacion cero forzada
  expect_true(any(panel$lgd == 0))
  expect_true(any(panel$lgd == 1))

  # mgcv::betar() con y en el borde exacto ajusta con una advertencia de
  # verosimilitud saturada (esperado, documentado), no con un error -- eso
  # es exactamente el "manejo correcto" que este test verifica.
  fit <- NULL
  expect_warning(
    fit <- fit_lgd_gam(panel),
    regexp = "saturated likelihood"
  )
  expect_false(is.null(fit))
})

test_that("las predicciones de fit_lgd_gam quedan estrictamente en (0,1) aun con bordes en el train", {
  set.seed(4)
  panel <- simulate_defaulted_loan_panel(macro_sintetico(), loans_per_year = 40)
  panel$lgd[1] <- 0
  panel$lgd[2] <- 1

  fit <- suppressWarnings(fit_lgd_gam(panel))
  preds <- predict(fit, type = "response")

  expect_false(anyNA(preds))
  expect_true(all(preds > 0))
  expect_true(all(preds < 1))  # el link logit nunca devuelve exactamente 0 o 1
})

test_that("fit_lgd_gam ajusta sin advertencias cuando el train es todo interior a (0,1)", {
  set.seed(6)
  panel <- simulate_defaulted_loan_panel(macro_sintetico(), loans_per_year = 60)
  panel_interior <- panel[panel$lgd > 0 & panel$lgd < 1, ]
  expect_gt(nrow(panel_interior), 30)

  expect_no_warning(fit_lgd_gam(panel_interior))
})
