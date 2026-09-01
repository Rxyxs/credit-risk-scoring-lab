#ifndef SCORE_ENGINE_H
#define SCORE_ENGINE_H

#ifdef _WIN32
#define SCOREAPI __declspec(dllexport)
#else
#define SCOREAPI
#endif

/* Puntaje de un solicitante: suma los puntos del bin correspondiente de
 * cada feature (lookup en points_table, tabla aplanada feature-mayor de
 * tamano n_features * max_bins) mas los puntos base del scorecard. */
SCOREAPI double score_one(
    const int* bin_indices,
    int n_features,
    const double* points_table,
    int max_bins,
    double base_points
);

/* Version en lote: bin_indices es una matriz n_rows x n_features aplanada
 * fila-mayor. Escribe n_rows puntajes en out_scores (ya reservado por el
 * caller). Este es el hot-path que se compara contra Python/NumPy. */
SCOREAPI void score_batch(
    const int* bin_indices,
    int n_rows,
    int n_features,
    const double* points_table,
    int max_bins,
    double base_points,
    double* out_scores
);

#endif
