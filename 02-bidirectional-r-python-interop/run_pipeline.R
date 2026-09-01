# Orquestador maestro: corre el lado Python completo, luego el lado R, luego el
# puente combinado -- todo como subprocesos independientes (Python via el
# interprete del venv, R via `Rscript`), por la misma razon documentada en
# chile-energy-grid-forecasting-r: un guard `if (sys.nframe() == 0)` (o el
# equivalente de "solo ejecutar como script principal") no se activa si el
# archivo se carga con source() en vez de ejecutarse como proceso de tope.
#
# Ejecutar desde la raiz del repositorio con:
#   Rscript run_pipeline.R

venv_python <- file.path(getwd(), ".venv", "Scripts", "python.exe")
rscript_bin <- file.path(R.home("bin"), "Rscript")

steps <- list(
  list(desc = "1/8 Python: limpieza de datos + credit scoring + SHAP", cmd = venv_python, args = c("-m", "python.orchestrator")),
  list(desc = "2/8 Python: descarga de indicadores macro reales de Chile (Banco Mundial)", cmd = venv_python, args = c("-m", "python.fetch_macro_data")),
  list(desc = "3/8 R: generacion de precios OHLC (GARCH)", cmd = rscript_bin, args = "r/market_data.R"),
  list(desc = "4/8 R: velas japonesas + indicadores tecnicos", cmd = rscript_bin, args = "r/candlestick_charts.R"),
  list(desc = "5/8 R: GARCH de volatilidad de mercado", cmd = rscript_bin, args = "r/volatility_garch.R"),
  list(desc = "6/8 R: calibracion empirica de LGD (Tobit + GAM sobre datos macro reales)", cmd = rscript_bin, args = "r/lgd_calibration.R"),
  list(desc = "7/8 Python->R (rpy2): stress test IFRS9/Basilea III de LGD estresada", cmd = venv_python, args = c("-m", "python.lgd_stress_test")),
  list(desc = "8/8 R->Python (reticulate): analisis combinado de riesgo", cmd = rscript_bin, args = "r/run_combined_analysis.R")
)

for (step in steps) {
  cat(sprintf("\n%s\n>> %s\n%s\n", strrep("=", 70), step$desc, strrep("=", 70)))
  status <- system2(step$cmd, args = step$args)
  if (status != 0) {
    stop(sprintf("Paso fallido: %s (codigo %d)", step$desc, status))
  }
}

cat("\nPipeline completo. Resultados en data/, output/models/, output/tables/, output/figures/\n")
