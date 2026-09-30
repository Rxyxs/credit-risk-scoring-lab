# Tests de las funciones puras de binning WOE/IV (01_woe_binning.R) y de
# escalamiento PDO (02_scorecard_model.R). Datos sinteticos generados on the
# fly -- no leen ni escriben nada en data/ ni outputs/.

source("../../R/01_woe_binning.R")
source("../../R/02_scorecard_model.R")

# --------------------------------------------------------- WOE / IV

test_that("compute_woe_table no divide por cero cuando un bin no tiene malos", {
  bin_id <- c(rep(1, 20), rep(2, 20))
  target <- c(rep(0, 20), rep(c(0, 1), 10))  # bin 1: 0 malos
  total_good <- sum(target == 0)
  total_bad <- sum(target == 1)

  tab <- compute_woe_table(bin_id, target, total_good, total_bad)

  expect_false(anyNA(tab$woe))
  expect_true(all(is.finite(tab$woe)))

  bin1 <- tab[tab$bin_id == 1, ]
  expect_equal(bin1$n_bad, 0)
  expect_equal(bin1$dist_bad, 0.5 / total_bad)  # piso de 0.5, nunca 0
})

test_that("compute_woe_table no divide por cero cuando un bin no tiene buenos", {
  bin_id <- c(rep(1, 20), rep(2, 20))
  target <- c(rep(1, 20), rep(c(0, 1), 10))  # bin 1: 0 buenos
  total_good <- sum(target == 0)
  total_bad <- sum(target == 1)

  tab <- compute_woe_table(bin_id, target, total_good, total_bad)

  expect_false(anyNA(tab$woe))
  expect_true(all(is.finite(tab$woe)))

  bin1 <- tab[tab$bin_id == 1, ]
  expect_equal(bin1$n_good, 0)
  expect_equal(bin1$dist_good, 0.5 / total_good)
})

test_that("iv_contrib es siempre no negativo (divergencia bien definida)", {
  set.seed(11)
  bin_id <- sample(1:4, 200, replace = TRUE)
  target <- rbinom(200, 1, 0.3)
  tab <- compute_woe_table(bin_id, target, sum(target == 0), sum(target == 1))
  expect_true(all(tab$iv_contrib >= -1e-9))
})

test_that("fit_numeric_bins no produce NA y cubre todo el rango de x", {
  set.seed(21)
  n <- 500
  x <- rnorm(n, mean = 40, sd = 10)
  target <- rbinom(n, 1, plogis((x - 40) / 10))

  fit <- fit_numeric_bins(x, target)

  expect_false(anyNA(fit$bin_id))
  expect_length(fit$bin_id, n)
  expect_identical(fit$breaks[1], -Inf)
  expect_identical(fit$breaks[length(fit$breaks)], Inf)
  expect_gte(length(unique(fit$bin_id)), MIN_BINS)
})

test_that("fit_numeric_bins fusiona hasta lograr WOE monotono", {
  set.seed(22)
  n <- 500
  x <- rnorm(n, mean = 40, sd = 10)
  target <- rbinom(n, 1, plogis((x - 40) / 10))

  fit <- fit_numeric_bins(x, target)
  woe_tab <- compute_woe_table(fit$bin_id, target, sum(target == 0), sum(target == 1))

  expect_true(is_monotonic(woe_tab$woe))
})

test_that("fit_numeric_bins funciona con una variable discreta de pocos valores", {
  set.seed(23)
  x <- sample(0:3, 300, replace = TRUE, prob = c(0.7, 0.15, 0.1, 0.05))
  target <- rbinom(300, 1, 0.1 + 0.2 * x)

  fit <- fit_numeric_bins(x, target)

  expect_false(anyNA(fit$bin_id))
  expect_length(fit$bin_id, 300)
})

test_that("fit_categorical_bins asigna un bin por categoria y no produce NA", {
  set.seed(24)
  x <- sample(c("A", "B", "C"), 200, replace = TRUE)
  fit <- fit_categorical_bins(x)

  expect_false(anyNA(fit$bin_id))
  expect_setequal(fit$levels, c("A", "B", "C"))
  expect_length(fit$bin_id, 200)
})

test_that("is_monotonic detecta series crecientes, decrecientes y no monotonas", {
  expect_true(is_monotonic(c(1, 2, 2, 3)))
  expect_true(is_monotonic(c(3, 2, 2, 1)))
  expect_false(is_monotonic(c(1, 3, 2)))
})

# ------------------------------------------------- escalamiento PDO

test_that("pdo_scaling reproduce Factor y Offset segun la formula documentada", {
  s <- pdo_scaling(intercept = -1.4, pdo = 20, score_ref = 600, odds_ref = 50)

  expect_equal(s$factor, 20 / log(2))
  expect_equal(s$offset, 600 - s$factor * log(50))
  expect_equal(s$base_points, s$offset - s$factor * (-1.4))
})

test_that("duplicar el odds sube el score en exactamente PDO puntos", {
  s <- pdo_scaling(intercept = -1.0, pdo = 20, score_ref = 600, odds_ref = 50)
  score_en <- function(odds) s$offset + s$factor * log(odds)

  for (odds in c(5, 25, 100, 400)) {
    expect_equal(score_en(2 * odds) - score_en(odds), 20, tolerance = 1e-9)
  }
})

test_that("el score en Odds_ref coincide exactamente con Score_ref", {
  s <- pdo_scaling(intercept = -1.0, pdo = 20, score_ref = 600, odds_ref = 50)
  score_en_ref <- s$offset + s$factor * log(50)
  expect_equal(score_en_ref, 600, tolerance = 1e-9)
})

test_that("base_points baja cuando el intercepto sube (mas riesgo base, menos puntos)", {
  s_bajo_riesgo <- pdo_scaling(intercept = -2.0)
  s_alto_riesgo <- pdo_scaling(intercept = 0.5)
  expect_lt(s_alto_riesgo$base_points, s_bajo_riesgo$base_points)
})

test_that("pdo_scaling respeta parametros PDO/Score_ref/Odds_ref no default", {
  s <- pdo_scaling(intercept = 0, pdo = 40, score_ref = 700, odds_ref = 20)
  expect_equal(s$factor, 40 / log(2))
  expect_equal(s$offset, 700 - s$factor * log(20))
})
