# Puente R <-> Python: usa reticulate para invocar el pipeline de credit scoring de
# Python DIRECTAMENTE desde R (no solo lectura de un CSV que Python ya escribio --
# interoperabilidad real, R ejecuta el entrenamiento de Python y recibe de vuelta
# un data.frame de pandas ya convertido) y lo combina con el regimen de volatilidad
# de mercado (GARCH, R) en una matriz de perdida esperada estresada.
#
# Encuadre de negocio ("wrong-way risk"): en episodios de alta volatilidad de
# mercado, las tasas de default tienden a subir tambien (recesion, desempleo,
# tasas de interes altas). Los multiplicadores de estres por regimen (abajo) son
# SUPUESTOS ilustrativos y documentados como tales -- no estan calibrados contra
# datos reales de ciclo crediticio -- pero la estructura del analisis (PD ajustada
# por escenario macro x LGD x EAD, agregada por banda de riesgo) es la forma
# estandar en que la gestion de riesgo bancario real combina riesgo de credito y
# riesgo de mercado.

library(reticulate)
library(dplyr)
library(ggplot2)

LGD_SUPUESTO <- 0.45  # Loss Given Default -- cifra tipica de industria para credito
                       # de consumo no garantizado (~40-50%), no calibrada localmente.
                       # Una LGD empiricamente calibrada (Tobit + GAM sobre el ciclo
                       # macro real de Chile) existe en r/lgd_calibration.R y se usa
                       # en el stress test IFRS9/Basilea III de python/lgd_stress_test.py
                       # (output/tables/ifrs9_ecl_stress_test.csv) -- este mapa de
                       # calor se deja con el supuesto simple original a proposito,
                       # como comparacion honesta de linea base.

STRESS_MULTIPLIER <- c(Baja = 0.85, Media = 1.00, Alta = 1.35)

setup_python_bridge <- function() {
  venv_python <- file.path(getwd(), ".venv", "Scripts", "python.exe")
  Sys.setenv(RETICULATE_PYTHON = venv_python)
  # source_python() usa parent.frame() como entorno destino por defecto -- llamado
  # dentro de esta funcion auxiliar, eso ataria las funciones de Python al entorno
  # local de setup_python_bridge(), que desaparece al retornar. Se fuerza
  # explicitamente el entorno global para que run_credit_scoring_pipeline() quede
  # disponible despues de que esta funcion retorne.
  reticulate::source_python("python/credit_scoring.py", envir = globalenv())
}

#' Asigna cada cliente del set de test a una banda de riesgo (quintiles de PD).
assign_risk_bands <- function(risk_scores_df) {
  risk_scores_df %>%
    mutate(
      banda_riesgo = cut(
        pd_xgb,
        breaks = quantile(pd_xgb, probs = seq(0, 1, 0.2)),
        include.lowest = TRUE,
        labels = c("Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto")
      )
    )
}

#' Construye la matriz banda_riesgo x regimen_volatilidad de perdida esperada
#' estresada: EL = PD_estresada * LGD * EAD, agregada por banda.
build_stressed_expected_loss_matrix <- function(risk_scores_df) {
  con_bandas <- assign_risk_bands(risk_scores_df)

  regimenes <- names(STRESS_MULTIPLIER)
  resultado <- lapply(regimenes, function(regimen) {
    con_bandas %>%
      mutate(
        pd_estresada = pmin(1, pd_xgb * STRESS_MULTIPLIER[[regimen]]),
        perdida_esperada_clp = pd_estresada * LGD_SUPUESTO * monto_solicitado_clp
      ) %>%
      group_by(banda_riesgo) %>%
      summarise(
        n_clientes = n(),
        pd_promedio_estresada = mean(pd_estresada),
        exposicion_total_clp = sum(monto_solicitado_clp),
        perdida_esperada_total_clp = sum(perdida_esperada_clp),
        tasa_perdida_esperada = perdida_esperada_total_clp / exposicion_total_clp,
        .groups = "drop"
      ) %>%
      mutate(regimen_volatilidad = factor(regimen, levels = regimenes))
  })

  bind_rows(resultado)
}

plot_expected_loss_heatmap <- function(matriz, path) {
  p <- matriz %>%
    ggplot(aes(x = regimen_volatilidad, y = banda_riesgo, fill = tasa_perdida_esperada)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = scales::percent(tasa_perdida_esperada, accuracy = 0.1)), size = 4, fontface = "bold") +
    scale_fill_gradient(low = "#F4EFE8", high = "#B3452D", labels = scales::percent, name = "Tasa de\npérdida esperada") +
    labs(
      title = "Pérdida esperada estresada: riesgo de crédito × régimen de volatilidad de mercado",
      subtitle = "PD (XGBoost, Python) ajustada por régimen GARCH (R) × LGD 45% -- multiplicadores de estrés ilustrativos",
      x = "Régimen de volatilidad de mercado (GARCH)", y = "Banda de riesgo crediticio (quintil de PD)"
    ) +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())

  ggsave(path, plot = p, width = 9, height = 6, dpi = 150)
  p
}

if (sys.nframe() == 0) {
  cat("=== 1/3 Invocando el pipeline de credit scoring de Python desde R (reticulate) ===\n")
  setup_python_bridge()
  python_output <- run_credit_scoring_pipeline()
  risk_scores_df <- python_output$risk_scores_df  # pandas DataFrame -> R data.frame automatico
  cat(sprintf("  %d clientes de test recibidos desde Python\n", nrow(risk_scores_df)))

  cat("\n=== 2/3 Combinando con régimen de volatilidad de mercado (GARCH, R) ===\n")
  matriz_perdida <- build_stressed_expected_loss_matrix(risk_scores_df)
  print(matriz_perdida)

  dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)
  dir.create("output/figures", showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(matriz_perdida, "output/tables/stressed_expected_loss_matrix.csv")

  cat("\n=== 3/3 Generando mapa de calor combinado ===\n")
  plot_expected_loss_heatmap(matriz_perdida, "output/figures/combined_risk_heatmap.png")
  cat("Guardado en output/figures/combined_risk_heatmap.png\n")
}
