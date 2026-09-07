"""Binning optimo por programacion dinamica, con WOE e IV.

El binning es la decision menos glamorosa y mas determinante de un
scorecard: define en que tramos se corta cada variable, y con eso fija
tanto el poder predictivo como la lectura que despues alguien tiene que
defender ante un comite. La practica habitual es cortar por deciles (rapido
pero ciego a la etiqueta) o dejar que un arbol elija (usa la etiqueta, pero
es un heuristico greedy sin garantia de optimalidad ni de monotonia).

Aca el problema se plantea como lo que es -- una particion de un eje
ordenado que maximiza el Information Value sujeto a restricciones -- y se
resuelve **exactamente** con programacion dinamica:

    maximizar  sum_bins IV(bin)
    sujeto a   a lo mas K bins
               cada bin con al menos una fraccion minima de la poblacion
               cada bin con un minimo de eventos (y de no-eventos)
               WOE monotono a lo largo de los bins (opcional)

La monotonia es lo que hace que la DP sea la herramienta correcta y no un
lujo: un greedy que corta donde mas gana localmente no puede garantizar que
la secuencia final de WOE sea monotona, y un scorecard con WOE no monotono
en la carga financiera es exactamente el hallazgo que hunde una validacion.

Definiciones (con correccion de Laplace para evitar log(0)):

    WOE(bin) = log( (malos_bin / malos_total) / (buenos_bin / buenos_total) )
    IV(bin)  = (malos_bin/malos_total - buenos_bin/buenos_total) * WOE(bin)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUAVIZADO = 0.5           # correccion de Laplace por bin
MAX_BINS_DEFAULT = 6
MIN_FRACCION_DEFAULT = 0.05
MIN_EVENTOS_DEFAULT = 25
N_PREBINS_DEFAULT = 100

# Lectura estandar de la industria para el Information Value total.
ESCALA_IV = [
    (0.02, "sin poder predictivo"),
    (0.10, "debil"),
    (0.30, "medio"),
    (0.50, "fuerte"),
    (float("inf"), "sospechoso (revisar fuga de informacion)"),
]


def interpretar_iv(iv: float) -> str:
    for umbral, etiqueta in ESCALA_IV:
        if iv < umbral:
            return etiqueta
    return ESCALA_IV[-1][1]


@dataclass
class ResultadoBinning:
    """Bins finales de una variable, con sus estadisticos."""
    variable: str
    cortes: list[float]                  # bordes internos (solo numericas)
    grupos: list[list] | None            # agrupacion de categorias (categoricas)
    tabla: pd.DataFrame
    iv: float
    monotono: bool
    metodo: str
    detalle: dict = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return len(self.tabla)


# ----------------------------------------------------------------------
# Estadisticos de un tramo
# ----------------------------------------------------------------------
def _prefijos(conteo_total: np.ndarray, conteo_malos: np.ndarray):
    """Sumas acumuladas para poder evaluar cualquier tramo en O(1)."""
    return (np.concatenate([[0], np.cumsum(conteo_total)]),
            np.concatenate([[0], np.cumsum(conteo_malos)]))


def woe_iv_tramo(n: np.ndarray, malos: np.ndarray, total_malos: float,
                 total_buenos: float) -> tuple[np.ndarray, np.ndarray]:
    """WOE e IV de tramos con conteos `n` y `malos` (vectorizado)."""
    malos = np.asarray(malos, dtype=float)
    buenos = np.asarray(n, dtype=float) - malos
    p_malos = (malos + SUAVIZADO) / (total_malos + 2 * SUAVIZADO)
    p_buenos = (buenos + SUAVIZADO) / (total_buenos + 2 * SUAVIZADO)
    woe = np.log(p_malos / p_buenos)
    iv = (p_malos - p_buenos) * woe
    return woe, iv


# ----------------------------------------------------------------------
# Binning optimo
# ----------------------------------------------------------------------
class OptimalBinner:
    """Encuentra la particion de IV maximo bajo restricciones, por DP.

    Parameters
    ----------
    max_bins : maximo de bins finales.
    min_fraccion : fraccion minima de la poblacion por bin.
    min_eventos : minimo de malos y de buenos por bin.
    monotono : si True, exige WOE monotono (prueba las dos direcciones y
        se queda con la mejor).
    n_prebins : granularidad del pre-binning por cuantiles. La DP es
        exacta *sobre esta grilla*: los cortes solo pueden caer en bordes
        de prebin, asi que la grilla fija cuan cerca se puede quedar del
        optimo continuo. Con 100 prebins el paso es de 1% de la poblacion
        y la solucion ya practicamente no se mueve al refinar mas.
    """

    def __init__(self, max_bins: int = MAX_BINS_DEFAULT,
                 min_fraccion: float = MIN_FRACCION_DEFAULT,
                 min_eventos: int = MIN_EVENTOS_DEFAULT,
                 monotono: bool = True,
                 n_prebins: int = N_PREBINS_DEFAULT):
        if max_bins < 2:
            raise ValueError("max_bins debe ser al menos 2")
        if not 0 < min_fraccion < 0.5:
            raise ValueError("min_fraccion debe estar en (0, 0.5)")
        self.max_bins = int(max_bins)
        self.min_fraccion = float(min_fraccion)
        self.min_eventos = int(min_eventos)
        self.monotono = bool(monotono)
        self.n_prebins = int(n_prebins)

    # ------------------------------------------------------------------
    def _prebin_numerico(self, x: np.ndarray, y: np.ndarray):
        """Agrupa por cuantiles y devuelve conteos por prebin ordenado."""
        cuantiles = np.linspace(0, 1, self.n_prebins + 1)[1:-1]
        bordes = np.unique(np.quantile(x, cuantiles))
        idx = np.searchsorted(bordes, x, side="right")
        m = bordes.size + 1
        n = np.bincount(idx, minlength=m).astype(float)
        malos = np.bincount(idx, weights=y, minlength=m).astype(float)
        # prebins vacios no aportan y complican la DP: se descartan
        vivos = n > 0
        return bordes, n[vivos], malos[vivos], vivos

    def _resolver_dp(self, n: np.ndarray, malos: np.ndarray, direccion: int
                     ) -> tuple[float, list[tuple[int, int]]]:
        """DP exacta sobre tramos contiguos.

        `direccion` es +1 (WOE creciente), -1 (decreciente) o 0 (libre).
        Devuelve (IV total, lista de tramos como pares (inicio, fin)).
        """
        m = n.size
        total = n.sum()
        total_malos = malos.sum()
        total_buenos = total - total_malos
        min_n = self.min_fraccion * total

        # Estadisticos de todos los tramos (i..j) de una vez.
        cum_n, cum_malos = _prefijos(n, malos)
        idx_i, idx_j = np.triu_indices(m)
        n_tramo = cum_n[idx_j + 1] - cum_n[idx_i]
        malos_tramo = cum_malos[idx_j + 1] - cum_malos[idx_i]

        W = np.full((m, m), np.nan)
        IV = np.full((m, m), -np.inf)
        VALIDO = np.zeros((m, m), dtype=bool)

        woe, iv = woe_iv_tramo(n_tramo, malos_tramo, total_malos, total_buenos)
        buenos_tramo = n_tramo - malos_tramo
        valido = (
            (n_tramo >= min_n)
            & (malos_tramo >= self.min_eventos)
            & (buenos_tramo >= self.min_eventos)
        )
        W[idx_i, idx_j] = woe
        IV[idx_i, idx_j] = np.where(valido, iv, -np.inf)
        VALIDO[idx_i, idx_j] = valido

        NEG = -np.inf
        # dp[k, i, j]: mejor IV cubriendo 0..j con k tramos, ultimo tramo (i..j)
        dp = np.full((self.max_bins + 1, m, m), NEG)
        padre = np.full((self.max_bins + 1, m, m), -1, dtype=int)

        for j in range(m):
            if VALIDO[0, j]:
                dp[1, 0, j] = IV[0, j]

        for k in range(2, self.max_bins + 1):
            for j in range(m):
                for i in range(1, j + 1):
                    if not VALIDO[i, j]:
                        continue
                    # Todos los predecesores posibles de una vez: el tramo
                    # anterior termina en i-1 y empieza en algun i2 < i.
                    previos = dp[k - 1, :i, i - 1]
                    if direccion > 0:
                        admisibles = W[i, j] > W[:i, i - 1]
                    elif direccion < 0:
                        admisibles = W[i, j] < W[:i, i - 1]
                    else:
                        admisibles = np.ones(i, dtype=bool)
                    candidatos = np.where(admisibles & (previos > NEG), previos, NEG)
                    mejor_i2 = int(np.argmax(candidatos))
                    if candidatos[mejor_i2] > NEG:
                        dp[k, i, j] = IV[i, j] + candidatos[mejor_i2]
                        padre[k, i, j] = mejor_i2

        mejor_valor, mejor_estado = NEG, None
        for k in range(1, self.max_bins + 1):
            for i in range(m):
                if dp[k, i, m - 1] > mejor_valor:
                    mejor_valor, mejor_estado = dp[k, i, m - 1], (k, i)
        if mejor_estado is None:
            return NEG, []

        # Reconstruccion hacia atras.
        k, i = mejor_estado
        j = m - 1
        tramos = []
        while k >= 1:
            tramos.append((i, j))
            if k == 1:
                break
            i2 = padre[k, i, j]
            k, j, i = k - 1, i - 1, i2
        return float(mejor_valor), list(reversed(tramos))

    # ------------------------------------------------------------------
    def fit_numerica(self, x, y, nombre: str = "x") -> ResultadoBinning:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size != y.size:
            raise ValueError("x e y deben tener el mismo largo")
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("y debe ser binaria (0/1)")

        bordes, n, malos, _ = self._prebin_numerico(x, y)

        direcciones = (1, -1) if self.monotono else (0,)
        mejor = None
        for direccion in direcciones:
            valor, tramos = self._resolver_dp(n, malos, direccion)
            if tramos and (mejor is None or valor > mejor[0]):
                mejor = (valor, tramos, direccion)
        if mejor is None:
            # sin particion factible: un solo bin con todo
            mejor = (0.0, [(0, n.size - 1)], 0)

        _, tramos, direccion = mejor
        cortes = [float(bordes[j]) for (_, j) in tramos[:-1] if j < bordes.size]
        tabla = self._tabla_desde_tramos(n, malos, tramos, bordes, nombre)
        return ResultadoBinning(
            variable=nombre, cortes=cortes, grupos=None, tabla=tabla,
            iv=float(tabla["iv"].sum()),
            monotono=bool(_es_monotona(tabla["woe"].to_numpy())),
            metodo="dp_optimo" + ("_monotono" if self.monotono else ""),
            detalle={"direccion": int(direccion), "n_prebins": int(n.size)},
        )

    def fit_categorica(self, x, y, nombre: str = "x") -> ResultadoBinning:
        """Categoricas: se ordenan por tasa de malos y se corre la misma DP.

        Ordenar por tasa de evento antes de agrupar es lo que convierte un
        problema combinatorio (todas las particiones posibles de un conjunto)
        en el mismo problema de tramos contiguos que resuelve la DP.
        """
        x = pd.Series(x).astype(str).to_numpy()
        y = np.asarray(y, dtype=float)
        df = pd.DataFrame({"cat": x, "y": y})
        g = df.groupby("cat")["y"].agg(["size", "sum"]).rename(
            columns={"size": "n", "sum": "malos"})
        g["tasa"] = g["malos"] / g["n"]
        g = g.sort_values("tasa")

        n = g["n"].to_numpy(float)
        malos = g["malos"].to_numpy(float)
        categorias = g.index.to_numpy()

        direcciones = (1, -1) if self.monotono else (0,)
        mejor = None
        for direccion in direcciones:
            valor, tramos = self._resolver_dp(n, malos, direccion)
            if tramos and (mejor is None or valor > mejor[0]):
                mejor = (valor, tramos, direccion)
        if mejor is None:
            mejor = (0.0, [(0, n.size - 1)], 0)

        _, tramos, direccion = mejor
        grupos = [list(categorias[i:j + 1]) for (i, j) in tramos]
        tabla = self._tabla_desde_tramos(n, malos, tramos, None, nombre, grupos)
        return ResultadoBinning(
            variable=nombre, cortes=[], grupos=grupos, tabla=tabla,
            iv=float(tabla["iv"].sum()),
            monotono=bool(_es_monotona(tabla["woe"].to_numpy())),
            metodo="dp_optimo_categorico",
            detalle={"direccion": int(direccion), "n_categorias": int(n.size)},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _tabla_desde_tramos(n, malos, tramos, bordes, nombre, grupos=None) -> pd.DataFrame:
        total_malos = float(malos.sum())
        total_buenos = float(n.sum() - malos.sum())
        filas = []
        for b, (i, j) in enumerate(tramos):
            n_bin = float(n[i:j + 1].sum())
            malos_bin = float(malos[i:j + 1].sum())
            woe, iv = woe_iv_tramo(np.array([n_bin]), np.array([malos_bin]),
                                   total_malos, total_buenos)
            if grupos is not None:
                etiqueta = " | ".join(map(str, grupos[b]))
            else:
                lo = "-inf" if i == 0 else f"{bordes[i - 1]:.4g}"
                hi = "+inf" if j >= bordes.size else f"{bordes[j]:.4g}"
                etiqueta = f"({lo}, {hi}]"
            filas.append({
                "variable": nombre,
                "bin": b,
                "etiqueta": etiqueta,
                "n": int(n_bin),
                "pct_poblacion": n_bin / float(n.sum()),
                "malos": int(malos_bin),
                "tasa_mala": malos_bin / n_bin,
                "woe": float(woe[0]),
                "iv": float(iv[0]),
            })
        return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# Metodos de referencia con los que comparar
# ----------------------------------------------------------------------
def binning_equifrecuente(x, y, n_bins: int = MAX_BINS_DEFAULT,
                          nombre: str = "x") -> ResultadoBinning:
    """Cortes por cuantiles: ignora la etiqueta por completo."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bordes = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return _resultado_desde_cortes(x, y, bordes, nombre, "equifrecuente")


def binning_arbol(x, y, n_bins: int = MAX_BINS_DEFAULT, min_fraccion: float = 0.05,
                  nombre: str = "x", seed: int = 42) -> ResultadoBinning:
    """Cortes de un arbol de decision: usa la etiqueta, pero es greedy."""
    from sklearn.tree import DecisionTreeClassifier

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    arbol = DecisionTreeClassifier(
        max_leaf_nodes=n_bins, min_samples_leaf=max(int(min_fraccion * x.size), 1),
        random_state=seed,
    ).fit(x.reshape(-1, 1), y)
    umbral = arbol.tree_.threshold[arbol.tree_.feature >= 0]
    bordes = np.unique(umbral)
    return _resultado_desde_cortes(x, y, bordes, nombre, "arbol")


def _resultado_desde_cortes(x, y, bordes, nombre, metodo) -> ResultadoBinning:
    idx = np.searchsorted(bordes, x, side="right")
    m = bordes.size + 1
    n = np.bincount(idx, minlength=m).astype(float)
    malos = np.bincount(idx, weights=y, minlength=m).astype(float)
    vivos = n > 0
    n, malos = n[vivos], malos[vivos]
    tramos = [(i, i) for i in range(n.size)]
    tabla = OptimalBinner._tabla_desde_tramos(n, malos, tramos, bordes, nombre)
    return ResultadoBinning(
        variable=nombre, cortes=[float(b) for b in bordes], grupos=None, tabla=tabla,
        iv=float(tabla["iv"].sum()),
        monotono=bool(_es_monotona(tabla["woe"].to_numpy())),
        metodo=metodo, detalle={},
    )


def _es_monotona(woe: np.ndarray) -> bool:
    d = np.diff(woe)
    return bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))


# ----------------------------------------------------------------------
# Aplicacion de un binning a datos nuevos
# ----------------------------------------------------------------------
def aplicar_binning(resultado: ResultadoBinning, x) -> np.ndarray:
    """Devuelve el WOE de cada observacion segun los bins ya ajustados."""
    woe = resultado.tabla["woe"].to_numpy(float)
    if resultado.grupos is not None:
        mapa = {cat: b for b, grupo in enumerate(resultado.grupos) for cat in grupo}
        idx = pd.Series(x).astype(str).map(mapa)
        # Una categoria no vista en train cae al bin de mayor poblacion.
        bin_refugio = int(resultado.tabla["n"].idxmax())
        return woe[idx.fillna(bin_refugio).astype(int).to_numpy()]

    cortes = np.asarray(resultado.cortes, dtype=float)
    idx = np.searchsorted(cortes, np.asarray(x, dtype=float), side="right")
    return woe[np.clip(idx, 0, woe.size - 1)]


def indice_bin(resultado: ResultadoBinning, x) -> np.ndarray:
    """Indice del bin al que cae cada observacion (para PSI/CSI)."""
    if resultado.grupos is not None:
        mapa = {cat: b for b, grupo in enumerate(resultado.grupos) for cat in grupo}
        bin_refugio = int(resultado.tabla["n"].idxmax())
        return pd.Series(x).astype(str).map(mapa).fillna(bin_refugio).astype(int).to_numpy()
    cortes = np.asarray(resultado.cortes, dtype=float)
    idx = np.searchsorted(cortes, np.asarray(x, dtype=float), side="right")
    return np.clip(idx, 0, len(resultado.tabla) - 1)
