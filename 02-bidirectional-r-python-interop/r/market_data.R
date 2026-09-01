# Genera precios OHLC (Open-High-Low-Close) sintéticos para un índice bursátil
# chileno ilustrativo, cuyos retornos siguen un proceso GARCH(1,1) explícito
# (misma técnica validada en chile-energy-grid-forecasting-r, aplicada ahora a su
# dominio clásico: retornos financieros, no volatilidad eólica).
#
# Por qué esto conecta con el riesgo de crédito: el riesgo de mercado y el riesgo
# de crédito están correlacionados en la práctica ("wrong-way risk") -- en
# episodios de alta volatilidad de mercado, las tasas de default también suben.
# El bridge/ usa el régimen de volatilidad de este índice para estresar la pérdida
# esperada de la cartera de crédito calculada en python/.

library(dplyr)

simulate_garch11_returns <- function(n, omega, alpha, beta, seed) {
  set.seed(seed)
  sigma2 <- numeric(n); eps <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta)
  eps[1] <- sqrt(sigma2[1]) * rnorm(1)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * eps[t - 1]^2 + beta * sigma2[t - 1]
    eps[t] <- sqrt(sigma2[t]) * rnorm(1)
  }
  list(returns = eps, sigma = sqrt(sigma2))
}

#' Genera una serie de precios OHLC diarios a partir de un proceso GARCH(1,1) de
#' retornos logaritmicos, con un pequeño ruido intradia fisicamente consistente
#' (High >= max(Open,Close), Low <= min(Open,Close)).
generate_ohlc_prices <- function(n_days = 750, precio_inicial = 5200, seed = 100) {
  garch <- simulate_garch11_returns(n_days, omega = 0.000006, alpha = 0.09, beta = 0.88, seed = seed)

  log_returns <- garch$returns + 0.0002  # leve deriva positiva (mercado alcista de largo plazo)
  precios_cierre <- precio_inicial * cumprod(1 + log_returns)

  set.seed(seed + 1)
  precios_apertura <- c(precio_inicial, head(precios_cierre, -1)) * (1 + rnorm(n_days, 0, 0.0015))

  rango_intradia <- abs(precios_cierre - precios_apertura) + precios_cierre * abs(rnorm(n_days, 0.004, 0.002))
  precios_max <- pmax(precios_apertura, precios_cierre) + rango_intradia * runif(n_days, 0.2, 0.6)
  precios_min <- pmin(precios_apertura, precios_cierre) - rango_intradia * runif(n_days, 0.2, 0.6)

  # Se piden suficientes dias calendario de sobra (factor 7/5 + margen) para que,
  # tras filtrar fines de semana, queden AL MENOS n_days dias habiles.
  dias_calendario_necesarios <- ceiling(n_days * 7 / 5) + 15
  fechas <- seq(Sys.Date() - dias_calendario_necesarios + 1, Sys.Date(), by = "day")
  fechas <- fechas[format(fechas, "%u") %in% as.character(1:5)]  # solo dias habiles
  fechas <- tail(fechas, n_days)

  tibble(
    fecha = fechas,
    open = round(precios_apertura, 2),
    high = round(precios_max, 2),
    low = round(precios_min, 2),
    close = round(precios_cierre, 2),
    volumen = round(runif(n_days, 2e6, 9e6)),
    log_return = round(log_returns, 6),
    volatilidad_condicional = round(garch$sigma, 6)
  )
}

if (sys.nframe() == 0) {
  ohlc <- generate_ohlc_prices()
  dir.create("data", showWarnings = FALSE)
  readr::write_csv(ohlc, "data/market_ohlc_synthetic.csv")
  cat(sprintf("Precios OHLC generados: %d dias -> data/market_ohlc_synthetic.csv\n", nrow(ohlc)))
  cat(sprintf("Precio inicial: %.1f | Precio final: %.1f | Retorno total: %.1f%%\n",
              ohlc$close[1], tail(ohlc$close, 1), 100 * (tail(ohlc$close, 1) / ohlc$open[1] - 1)))
}
