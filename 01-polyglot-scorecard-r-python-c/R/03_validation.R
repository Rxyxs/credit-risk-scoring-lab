# Validacion del scorecard: AUC/Gini, estadistico KS, y PSI (Population
# Stability Index) -- las tres metricas estandar de gobierno de modelos de
# riesgo de credito en banca.
#
# PSI se calcula dos veces para que el numero sea interpretable por
# contraste: (1) test vs. train, que deberia salir cercano a 0 porque es
# el mismo proceso generador con un split aleatorio estratificado -- eso
# es la prueba de que el metodo no arroja falsos positivos de drift; y
# (2) una poblacion "futura" simulada con sobre-representacion de perfiles
# de mayor riesgo (informal, DTI alto), que deberia mostrar PSI elevado --
# la prueba de que el metodo SI detecta drift real cuando existe.
#
# PSI = suma_bins[ (%actual - %esperado) * ln(%actual / %esperado) ]
# Umbrales estandar de la industria: <0.10 sin cambio significativo,
# 0.10-0.25 cambio moderado (investigar), >0.25 cambio significativo
# (recalibrar).
#
# Ejecutar desde la raiz del repo: Rscript R/03_validation.R

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

N_PSI_BINS <- 10

compute_auc <- function(score, target) {
  # AUC via el estadistico de Mann-Whitney U (equivalente exacto a la
  # curva ROC, sin depender de un paquete externo): score mas alto debe
  # significar MENOS riesgo, por lo que se calcula el rango sobre -score.
  r <- rank(-score)
  n_bad <- sum(target == 1)
  n_good <- sum(target == 0)
  sum_rank_bad <- sum(r[target == 1])
  u <- sum_rank_bad - n_bad * (n_bad + 1) / 2
  u / (n_bad * n_good)
}

compute_psi <- function(score_baseline, score_actual, breaks) {
  cut_baseline <- cut(score_baseline, breaks = breaks, include.lowest = TRUE)
  cut_actual <- cut(score_actual, breaks = breaks, include.lowest = TRUE)

  pct_baseline <- pmax(as.numeric(table(cut_baseline)) / length(score_baseline), 1e-4)
  pct_actual <- pmax(as.numeric(table(cut_actual)) / length(score_actual), 1e-4)

  psi_contrib <- (pct_actual - pct_baseline) * log(pct_actual / pct_baseline)
  list(
    psi = sum(psi_contrib),
    table = data.frame(
      bin = levels(cut_baseline),
      pct_baseline = pct_baseline,
      pct_actual = pct_actual,
      psi_contrib = psi_contrib
    )
  )
}

simulate_drifted_population <- function(data, n = 5000, seed = 42) {
  set.seed(seed)
  weight <- ifelse(data$tipo_contrato == "informal", 3.0, 1.0) *
    ifelse(data$dti > 1.5, 2.0, 1.0)
  idx <- sample(seq_len(nrow(data)), size = n, replace = TRUE, prob = weight / sum(weight))
  data[idx, ]
}

main <- function() {
  scored <- read.csv("data/processed/applicants_scored.csv", stringsAsFactors = FALSE)
  clean <- read.csv("data/processed/applicants_clean.csv", stringsAsFactors = FALSE)

  test <- scored[scored$split == "test", ]
  train <- scored[scored$split == "train", ]

  auc <- compute_auc(test$score, test$default_12m)
  gini <- 2 * auc - 1
  ks <- ks.test(test$score[test$default_12m == 0], test$score[test$default_12m == 1])

  psi_breaks <- unique(quantile(train$score, probs = seq(0, 1, length.out = N_PSI_BINS + 1)))
  psi_breaks[1] <- -Inf
  psi_breaks[length(psi_breaks)] <- Inf

  psi_test <- compute_psi(train$score, test$score, psi_breaks)

  drifted_ids <- simulate_drifted_population(clean, n = 5000)$applicant_id
  drifted_scores <- scored$score[match(drifted_ids, scored$applicant_id)]
  psi_drifted <- compute_psi(train$score, drifted_scores, psi_breaks)

  dir.create("outputs/reports", recursive = TRUE, showWarnings = FALSE)

  metrics <- list(
    n_test = nrow(test),
    n_train = nrow(train),
    auc = auc,
    gini = gini,
    ks_statistic = unname(ks$statistic),
    ks_p_value = ks$p.value,
    psi_test_vs_train = psi_test$psi,
    psi_drifted_vs_train = psi_drifted$psi
  )
  write_json(metrics, "outputs/reports/scorecard_validation_metrics.json", auto_unbox = TRUE, digits = 6)

  write.csv(psi_test$table, "outputs/reports/psi_bins_test.csv", row.names = FALSE)
  write.csv(psi_drifted$table, "outputs/reports/psi_bins_drifted.csv", row.names = FALSE)
  write.csv(scored, "outputs/reports/score_distribution.csv", row.names = FALSE)

  cat(sprintf("AUC (test): %.4f | Gini: %.4f | KS: %.4f (p=%.2e)\n", auc, gini, ks$statistic, ks$p.value))
  cat(sprintf("PSI test vs train: %.4f (esperado: bajo, mismo proceso generador)\n", psi_test$psi))
  cat(sprintf("PSI poblacion drifted vs train: %.4f (esperado: elevado, mezcla de riesgo distinta)\n", psi_drifted$psi))
}

if (sys.nframe() == 0) {
  main()
}
