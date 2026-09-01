# Gráficos de velas japonesas (candlestick) e indicadores técnicos clásicos
# (medias móviles, Bandas de Bollinger, RSI) via quantmod/TTR -- las herramientas
# estándar de R para análisis técnico de mercado.

library(quantmod)
library(TTR)
library(xts)
library(dplyr)

#' Convierte el tibble OHLC a un objeto xts (el formato que espera quantmod).
to_xts <- function(ohlc_df) {
  xts(
    ohlc_df[, c("open", "high", "low", "close", "volumen")],
    order.by = ohlc_df$fecha
  ) |> setNames(c("Open", "High", "Low", "Close", "Volume"))
}

#' Grafico de velas con medias moviles y Bandas de Bollinger superpuestas (ultimos
#' `n_recientes` dias, para que las velas individuales sean legibles).
plot_candlestick_with_indicators <- function(ohlc_xts, path, n_recientes = 120) {
  serie <- tail(ohlc_xts, n_recientes)

  png(path, width = 1400, height = 900, res = 150)
  chartSeries(
    serie,
    name = "Índice bursátil sintético -- velas japonesas (120 días)",
    theme = chartTheme("white"),
    TA = "addBBands(); addSMA(20, col='blue'); addSMA(50, col='darkorange'); addRSI()"
  )
  dev.off()
  invisible(NULL)
}

#' Calcula indicadores tecnicos como columnas adicionales (para tablas/analisis,
#' no solo para el grafico).
compute_technical_indicators <- function(ohlc_df) {
  close <- ohlc_df$close
  ohlc_df %>%
    mutate(
      sma_20 = as.numeric(SMA(close, n = 20)),
      sma_50 = as.numeric(SMA(close, n = 50)),
      rsi_14 = as.numeric(RSI(close, n = 14)),
      bb_upper = as.numeric(BBands(close, n = 20)[, "up"]),
      bb_lower = as.numeric(BBands(close, n = 20)[, "dn"])
    )
}

if (sys.nframe() == 0) {
  ohlc_df <- readr::read_csv("data/market_ohlc_synthetic.csv", show_col_types = FALSE)
  ohlc_xts <- to_xts(ohlc_df)

  dir.create("output/figures", showWarnings = FALSE, recursive = TRUE)
  plot_candlestick_with_indicators(ohlc_xts, "output/figures/market_candlestick.png")
  cat("Grafico de velas guardado en output/figures/market_candlestick.png\n")

  con_indicadores <- compute_technical_indicators(ohlc_df)
  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(con_indicadores, "output/tables/market_technical_indicators.csv")
  cat("Indicadores tecnicos guardados en output/tables/market_technical_indicators.csv\n")
}
