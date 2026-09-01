"""Puente ctypes hacia el motor de scoring compilado en C
(outputs/models/score_engine.dll). Convierte arrays NumPy al layout de
memoria que espera la firma C (int32 fila-mayor para bin_indices, float64
para la tabla de puntos) y expone una funcion Python de alto nivel.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

DLL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "models" / "score_engine.dll"


class ScoreEngine:
    """Wrapper del motor de scoring en C. Lanza `FileNotFoundError` con un
    mensaje claro si la DLL no fue compilada (ver c/build.ps1)."""

    def __init__(self, dll_path: Path = DLL_PATH):
        if not dll_path.exists():
            raise FileNotFoundError(
                f"No se encontro {dll_path}. Compila el motor primero: "
                f"powershell -File c/build.ps1"
            )
        self._lib = ctypes.CDLL(str(dll_path))
        self._lib.score_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.score_batch.restype = None

    def score_batch(
        self,
        bin_indices: np.ndarray,
        points_table: np.ndarray,
        base_points: float,
    ) -> np.ndarray:
        """bin_indices: (n_rows, n_features) int32. points_table: (n_features, max_bins) float64."""
        bin_indices = np.ascontiguousarray(bin_indices, dtype=np.int32)
        points_table = np.ascontiguousarray(points_table, dtype=np.float64)
        n_rows, n_features = bin_indices.shape
        max_bins = points_table.shape[1]

        out_scores = np.empty(n_rows, dtype=np.float64)

        self._lib.score_batch(
            bin_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(n_rows),
            ctypes.c_int32(n_features),
            points_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(max_bins),
            ctypes.c_double(base_points),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out_scores


def score_batch_numpy(bin_indices: np.ndarray, points_table: np.ndarray, base_points: float) -> np.ndarray:
    """Referencia vectorizada en NumPy (misma logica, sin C)."""
    n_rows, n_features = bin_indices.shape
    feature_idx = np.arange(n_features)
    picked = points_table[feature_idx, bin_indices]  # (n_rows, n_features) via broadcasting
    return picked.sum(axis=1) + base_points


def score_batch_python(bin_indices: np.ndarray, points_table: np.ndarray, base_points: float) -> np.ndarray:
    """Referencia en Python puro (loop anidado, sin vectorizar) -- el
    'strawman' honesto contra el que C deberia ganar por un margen grande."""
    n_rows, n_features = bin_indices.shape
    out = np.empty(n_rows, dtype=np.float64)
    for row in range(n_rows):
        total = base_points
        for f in range(n_features):
            total += points_table[f, bin_indices[row, f]]
        out[row] = total
    return out
