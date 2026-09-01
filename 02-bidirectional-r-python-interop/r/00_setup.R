# Instala (si falta) el set de paquetes de R usado en este proyecto.
# Ejecutar una vez con: Rscript r/00_setup.R

user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c(
  "dplyr", "readr", "ggplot2", "scales",   # manipulacion y graficos
  "quantmod", "TTR", "xts",                # velas japonesas e indicadores tecnicos
  "rugarch",                               # GARCH de volatilidad
  "reticulate"                             # puente hacia Python
)

missing <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing) > 0) {
  message("Instalando paquetes faltantes: ", paste(missing, collapse = ", "))
  install.packages(missing, lib = user_lib, repos = "https://cloud.r-project.org")
} else {
  message("Todos los paquetes requeridos ya estan instalados.")
}

message("Setup completo. Version de R: ", R.version.string)
