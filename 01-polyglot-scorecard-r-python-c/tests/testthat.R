# Runner de la suite testthat de esta tecnica.
# Ejecutar desde la raiz de la tecnica: Rscript tests/testthat.R

library(testthat)

test_dir("tests/testthat", stop_on_failure = TRUE)
