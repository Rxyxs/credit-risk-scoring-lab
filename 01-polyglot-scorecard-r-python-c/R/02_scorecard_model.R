# Scorecard de puntos: regresion logistica sobre variables WOE + conversion
# a puntos via el metodo PDO (Points to Double the Odds), el estandar de la
# industria bancaria para transformar un modelo logistico en una cartilla
# de puntos interpretable (Siddiqi, "Credit Risk Scorecards").
#
# Seleccion de variables: se descartan features con IV < 0.02 (regla
# estandar: por debajo de eso el poder predictivo se considera
# practicamente nulo) antes de ajustar el modelo, no despues.
#
# Score = Offset + Factor * ln(odds_bueno/malo)
# Factor = PDO / ln(2) ;  Offset = Score_ref - Factor * ln(Odds_ref)
# Puntos_feature,bin = -Factor * beta_feature * WOE_feature,bin
# Score_total = Base_Points + suma_features( Puntos_feature,bin_del_solicitante )
#
# Ejecutar desde la raiz del repo: Rscript R/02_scorecard_model.R

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

IV_THRESHOLD <- 0.02
SCORE_REF <- 600
ODDS_REF <- 50    # 50 buenos : 1 malo, en el score de referencia
PDO <- 20         # puntos para duplicar el odds

main <- function() {
  binned <- read.csv("data/processed/applicants_woe_binned.csv", stringsAsFactors = FALSE)
  iv_summary <- read.csv("outputs/reports/iv_summary.csv", stringsAsFactors = FALSE)

  selected_features <- iv_summary$feature[iv_summary$iv >= IV_THRESHOLD]
  cat(sprintf("Features seleccionadas (IV >= %.2f): %s\n", IV_THRESHOLD, paste(selected_features, collapse = ", ")))
  dropped <- setdiff(iv_summary$feature, selected_features)
  if (length(dropped) > 0) {
    cat(sprintf("Features descartadas (IV < %.2f): %s\n", IV_THRESHOLD, paste(dropped, collapse = ", ")))
  }

  woe_cols <- paste0(selected_features, "_woe")
  bin_cols <- paste0(selected_features, "_bin")

  train <- binned[binned$split == "train", ]
  formula_str <- paste("default_12m ~", paste(woe_cols, collapse = " + "))
  model <- glm(as.formula(formula_str), data = train, family = binomial())

  coefs <- coef(model)
  coef_table <- data.frame(
    feature = c("(intercept)", selected_features),
    coefficient = as.numeric(coefs),
    row.names = NULL
  )
  print(summary(model))

  bad_sign <- selected_features[coefs[woe_cols] > 0]
  if (length(bad_sign) > 0) {
    warning(sprintf(
      "Coeficiente con signo contrario al esperado (WOE mas alto deberia bajar el riesgo): %s",
      paste(bad_sign, collapse = ", ")
    ))
  }

  factor_ <- PDO / log(2)
  offset_ <- SCORE_REF - factor_ * log(ODDS_REF)
  base_points <- offset_ - factor_ * coefs["(Intercept)"]

  woe_bins <- read.csv("outputs/reports/woe_bins.csv", stringsAsFactors = FALSE)
  points_rows <- list()
  for (feat in selected_features) {
    beta <- coefs[paste0(feat, "_woe")]
    feat_bins <- woe_bins %>% filter(feature == feat) %>% arrange(bin_id)
    points_rows[[feat]] <- data.frame(
      feature = feat,
      bin_id = feat_bins$bin_id - 1L,   # 0-indexado para el motor en C
      points = -factor_ * beta * feat_bins$woe
    )
  }
  points_table <- bind_rows(points_rows)

  dir.create("outputs/reports", recursive = TRUE, showWarnings = FALSE)
  write.csv(coef_table, "outputs/reports/scorecard_coefficients.csv", row.names = FALSE)
  write.csv(points_table, "outputs/reports/scorecard_points_table.csv", row.names = FALSE)

  max_bins <- max(sapply(selected_features, function(f) sum(points_table$feature == f)))
  meta <- list(
    feature_order = selected_features,
    max_bins = max_bins,
    base_points = unname(base_points),
    factor = factor_,
    offset = offset_,
    score_ref = SCORE_REF,
    odds_ref = ODDS_REF,
    pdo = PDO
  )
  write_json(meta, "outputs/reports/scorecard_meta.json", auto_unbox = TRUE, digits = 10)

  # Aplicar el scorecard a TODOS los solicitantes (train + test) para dejar
  # el score y la PD implicita listos para validacion (03) y benchmark (C).
  points_lookup <- setNames(points_table$points, paste(points_table$feature, points_table$bin_id, sep = "_"))
  score <- rep(unname(base_points), nrow(binned))
  for (feat in selected_features) {
    bin_ids <- binned[[paste0(feat, "_bin")]]
    key <- paste(feat, bin_ids, sep = "_")
    score <- score + unname(points_lookup[key])
  }
  log_odds_good_bad <- (score - offset_) / factor_
  pd_estimate <- 1 / (1 + exp(log_odds_good_bad))

  scored <- data.frame(
    applicant_id = binned$applicant_id,
    split = binned$split,
    default_12m = binned$default_12m,
    score = score,
    pd_estimate = pd_estimate
  )
  write.csv(scored, "data/processed/applicants_scored.csv", row.names = FALSE)

  cat(sprintf("\nScorecard: base_points=%.1f, factor=%.2f, offset=%.1f\n", base_points, factor_, offset_))
  cat(sprintf("Score aplicado a %d solicitantes (train + test).\n", nrow(scored)))
  cat(sprintf("Score range: [%.0f, %.0f], media=%.0f\n", min(score), max(score), mean(score)))
}

if (sys.nframe() == 0) {
  main()
}
