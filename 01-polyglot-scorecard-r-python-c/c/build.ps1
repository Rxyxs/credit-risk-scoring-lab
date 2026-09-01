# Compila el motor de scoring con MSVC (cl.exe): una DLL para que Python la
# cargue via ctypes, y un ejecutable standalone para el benchmark en C
# puro. Requiere Visual Studio Build Tools / Visual Studio con el
# workload de C++ instalado.
#
# Ejecutar desde la raiz del repo:  powershell -File c\build.ps1

$ErrorActionPreference = "Stop"

$vcvarsCandidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) {
    throw "No se encontro vcvars64.bat. Instala Visual Studio Build Tools (workload 'Desktop development with C++')."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$cDir = Join-Path $repoRoot "c"
$outDir = Join-Path $repoRoot "outputs\models"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Usando vcvars64.bat: $vcvars"

$buildCmd = "call `"$vcvars`" && cd /d `"$cDir`" && " +
    "cl /nologo /O2 /LD /Fe:`"$outDir\score_engine.dll`" score_engine.c && " +
    "cl /nologo /O2 /Fe:`"$outDir\score_bench.exe`" score_engine.c bench_main.c"

cmd /c $buildCmd
if ($LASTEXITCODE -ne 0) {
    throw "Compilacion fallida (codigo $LASTEXITCODE)"
}

Write-Host "Compilado OK -> $outDir\score_engine.dll, $outDir\score_bench.exe"
