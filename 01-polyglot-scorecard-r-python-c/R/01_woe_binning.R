# Binning WOE (Weight of Evidence) + IV (Information Value).
#
# Para cada feature numerica: bins iniciales por deciles, luego fusion
# greedy de bins adyacentes hasta lograr WOE monotono (o casi) y que
# ningun bin tenga menos del 1% de las observaciones. Para categoricas:
# cada categoria es su propio bin. El binning se ajusta SOLO sobre train
# (nunca test), y los bordes resultantes se aplican despues a ambos splits.
#
# WOE_bin = ln( %buenos_bin / %malos_bin )   (bueno = default_12m == 0)
# IV_feature = suma_bins[ (%buenos_bin - %malos_bin) * WOE_bin ]
#
# Ejecutar desde la raiz del repo: Rscript R/01_woe_binning.R

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

MIN_BIN_FRAC <- 0.01
MIN_BINS <- 3
N_INITIAL_BINS <- 10

NUMERIC_FEATURES <- c(
  "edad", "renta_liquida", "antiguedad_laboral_meses",
  "n_productos_activos", "deuda_total", "dti", "n_morosidad_reportes"
)
CATEGORICAL_FEATURES <- c("tipo_contrato", "region")

compute_woe_table <- function(bin_id, target, total_good, total_bad) {
  df <- data.frame(bin_id = bin_id, target = target)
  df %>%
    group_by(bin_id) %>%
    summarise(
      n = n(),
      n_bad = sum(target == 1),
      n_good = sum(target == 0),
      .groups = "drop"
    ) %>%
    mutate(
      dist_good = pmax(n_good, 0.5) / total_good,
      dist_bad = pmax(n_bad, 0.5) / total_bad,
      woe = log(dist_good / dist_bad),
      iv_contrib = (dist_good - dist_bad) * woe
    ) %>%
    arrange(bin_id)
}

is_monotonic <- function(x) {
  all(diff(x) >= -1e-9) || all(diff(x) <= 1e-9)
}

fit_numeric_bins <- function(x, target) {
  total_good <- sum(target == 0)
  total_bad <- sum(target == 1)
  n <- length(x)

  unique_vals <- sort(unique(x))
  if (length(unique_vals) <= N_INITIAL_BINS) {
    # Variable discreta con pocos valores (p.ej. conteos muy sesgados como
    # n_morosidad_reportes, donde >80% de la masa esta en 0): los cuantiles
    # no tienen resolucion para separarlos, asi que cada valor arranca como
    # su propio bin y la fusion greedy decide despues si conviene juntarlos.
    midpoints <- (unique_vals[-length(unique_vals)] + unique_vals[-1]) / 2
    breaks <- c(-Inf, midpoints, Inf)
  } else {
    breaks <- unique(quantile(x, probs = seq(0, 1, length.out = N_INITIAL_BINS + 1), na.rm = TRUE))
    breaks[1] <- -Inf
    breaks[length(breaks)] <- Inf
  }
  bin_id <- as.integer(cut(x, breaks = breaks, include.lowest = TRUE, labels = FALSE))

  repeat {
    woe_tab <- compute_woe_table(bin_id, target, total_good, total_bad)
    n_bins <- nrow(woe_tab)
    small_bin <- which(woe_tab$n < MIN_BIN_FRAC * n)
    monotonic <- is_monotonic(woe_tab$woe)

    if (n_bins <= MIN_BINS || (monotonic && length(small_bin) == 0)) break

    if (length(small_bin) > 0) {
      merge_at <- small_bin[1]
    } else {
      woe_diffs <- abs(diff(woe_tab$woe))
      merge_at <- which.min(woe_diffs) + 1L
    }
    merge_lo <- max(1L, merge_at - 1L)
    drop_break <- breaks[merge_lo + 1L]
    breaks <- breaks[breaks != drop_break]
    bin_id <- as.integer(cut(x, breaks = breaks, include.lowest = TRUE, labels = FALSE))
  }

  list(breaks = breaks, bin_id = bin_id)
}

fit_categorical_bins <- function(x) {
  levels_sorted <- sort(unique(x))
  bin_id <- match(x, levels_sorted)
  list(levels = levels_sorted, bin_id = bin_id)
}

main <- function() {
  data <- read.csv("data/processed/applicants_clean.csv", stringsAsFactors = FALSE)
  train <- data[data$split == "train", ]
  target_train <- train$default_12m
  total_good <- sum(target_train == 0)
  total_bad <- sum(target_train == 1)

  all_woe_rows <- list()
  iv_rows <- list()
  bin_edges <- list()
  binned_full <- data.frame(applicant_id = data$applicant_id, split = data$split, default_12m = data$default_12m)

  for (feat in NUMERIC_FEATURES) {
    fit <- fit_numeric_bins(train[[feat]], target_train)
    woe_tab <- compute_woe_table(fit$bin_id, target_train, total_good, total_bad)
    iv_rows[[feat]] <- data.frame(feature = feat, iv = sum(woe_tab$iv_contrib), n_bins = nrow(woe_tab), type = "numeric")
    woe_tab$feature <- feat
    all_woe_rows[[feat]] <- woe_tab
    bin_edges[[feat]] <- list(type = "numeric", breaks = fit$breaks)

    bin_id_full <- as.integer(cut(data[[feat]], breaks = fit$breaks, include.lowest = TRUE, labels = FALSE))
    binned_full[[paste0(feat, "_bin")]] <- bin_id_full - 1L  # 0-indexado para el motor en C
    woe_lookup <- setNames(woe_tab$woe, woe_tab$bin_id)
    binned_full[[paste0(feat, "_woe")]] <- unname(woe_lookup[as.character(bin_id_full)])
  }

  for (feat in CATEGORICAL_FEATURES) {
    fit <- fit_categorical_bins(train[[feat]])
    woe_tab <- compute_woe_table(fit$bin_id, target_train, total_good, total_bad)
    iv_rows[[feat]] <- data.frame(feature = feat, iv = sum(woe_tab$iv_contrib), n_bins = nrow(woe_tab), type = "categorical")
    woe_tab$feature <- feat
    all_woe_rows[[feat]] <- woe_tab
    bin_edges[[feat]] <- list(type = "categorical", levels = fit$levels)

    bin_id_full <- match(data[[feat]], fit$levels)
    bin_id_full[is.na(bin_id_full)] <- length(fit$levels)  # categoria no vista -> ultimo bin
    binned_full[[paste0(feat, "_bin")]] <- bin_id_full - 1L
    woe_lookup <- setNames(woe_tab$woe, woe_tab$bin_id)
    binned_full[[paste0(feat, "_woe")]] <- unname(woe_lookup[as.character(bin_id_full)])
  }

  dir.create("outputs/reports", recursive = TRUE, showWarnings = FALSE)
  dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

  woe_table <- bind_rows(all_woe_rows)
  write.csv(woe_table, "outputs/reports/woe_bins.csv", row.names = FALSE)

  iv_summary <- bind_rows(iv_rows) %>% arrange(desc(iv))
  write.csv(iv_summary, "outputs/reports/iv_summary.csv", row.names = FALSE)

  write_json(bin_edges, "outputs/reports/woe_bin_edges.json", auto_unbox = TRUE, digits = 10)

  write.csv(binned_full, "data/processed/applicants_woe_binned.csv", row.names = FALSE)

  cat("Binning WOE completo.\n")
  cat(sprintf("IV por feature (train, n=%d, defaults=%d):\n", nrow(train), total_bad))
  print(iv_summary)
}

if (sys.nframe() == 0) {
  main()
}
