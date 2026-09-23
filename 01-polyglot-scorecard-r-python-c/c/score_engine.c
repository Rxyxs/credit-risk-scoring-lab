#include "score_engine.h"
#include <math.h>
#include <stddef.h>

/* n_features == 0 es un caso valido y no un error (aplicante sin features
 * seleccionadas, score = base_points); n_features < 0, max_bins <= 0,
 * punteros NULL cuando hay algo que leer, o un bin fuera de
 * [0, max_bins) son entradas invalidas y devuelven NAN en vez de leer
 * fuera de los limites de points_table. */
double score_one(
    const int* bin_indices,
    int n_features,
    const double* points_table,
    int max_bins,
    double base_points
) {
    if (n_features < 0 || max_bins <= 0) {
        return NAN;
    }
    if (n_features > 0 && (bin_indices == NULL || points_table == NULL)) {
        return NAN;
    }
    double total = base_points;
    for (int f = 0; f < n_features; f++) {
        int bin = bin_indices[f];
        if (bin < 0 || bin >= max_bins) {
            return NAN;
        }
        total += points_table[f * max_bins + bin];
    }
    return total;
}

void score_batch(
    const int* bin_indices,
    int n_rows,
    int n_features,
    const double* points_table,
    int max_bins,
    double base_points,
    double* out_scores
) {
    if (out_scores == NULL || n_rows <= 0) {
        return;
    }
    if (n_features > 0 && (bin_indices == NULL || points_table == NULL)) {
        for (int row = 0; row < n_rows; row++) {
            out_scores[row] = NAN;
        }
        return;
    }
    for (int row = 0; row < n_rows; row++) {
        const int* row_bins = bin_indices ? bin_indices + (size_t)row * n_features : NULL;
        out_scores[row] = score_one(row_bins, n_features, points_table, max_bins, base_points);
    }
}
