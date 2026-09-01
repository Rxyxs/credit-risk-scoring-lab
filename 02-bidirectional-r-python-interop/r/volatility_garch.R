# Ajusta un GARCH(1,1) sobre los retornos logarítmicos del índice y clasifica cada
# día en un régimen de volatilidad (Baja/Media/Alta) por terciles de la volatilidad
# condicional estimada. Ese régimen es el insumo que el bridge/ usa para estresar
# la pérdida esperada de la cartera de crédito.

library(rugarch)
library(dplyr)

fit_market_garch <- function(log_returns) {
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"  # t-student: colas mas pesadas, estandar en retornos financieros
  )
  ugarchfit(spec = spec, data = log_returns, solver = "hybrid")
}

#' Clasifica la volatilidad condicional en terciles (Baja/Media/Alta).
classify_volatility_regime <- function(sigma_series) {
  cortes <- quantile(sigma_series, probs = c(1 / 3, 2 / 3), na.rm = TRUE)
  cut(
    sigma_series,
    breaks = c(-Inf, cortes[1], cortes[2], Inf),
    labels = c("Baja", "Media", "Alta")
  )
}

if (sys.nframe() == 0) {
  ohlc_df <- readr::read_csv("data/market_ohlc_synthetic.csv", show_col_types = FALSE)

  garch_fit <- fit_market_garch(ohlc_df$log_return)
  cat("Coeficientes GARCH(1,1) estimados (retornos de mercado):\n")
  print(rugarch::coef(garch_fit))

  sigma_estimada <- as.numeric(rugarch::sigma(garch_fit))
  regimen <- classify_volatility_regime(sigma_estimada)

  resultado <- ohlc_df %>%
    mutate(volatilidad_estimada_garch = sigma_estimada, regimen_volatilidad = regimen)

  dir.create("output/models", showWarnings = FALSE, recursive = TRUE)
  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)
  saveRDS(garch_fit, "output/models/market_garch_fit.rds")
  readr::write_csv(resultado, "output/tables/market_volatility_regimes.csv")

  cat("\nDistribucion de regimenes de volatilidad:\n")
  print(table(regimen))

  # Pronostico de volatilidad a 20 dias (aprox. 1 mes bursatil)
  fc <- rugarch::ugarchforecast(garch_fit, n.ahead = 20)
  sigma_fc <- as.numeric(rugarch::sigma(fc))
  cat(sprintf("\nVolatilidad condicional pronosticada: %.5f (h+1) -> %.5f (h+20)\n",
              sigma_fc[1], sigma_fc[20]))
}
