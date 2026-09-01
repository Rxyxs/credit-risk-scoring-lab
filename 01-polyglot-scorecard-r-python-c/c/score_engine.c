#include "score_engine.h"

double score_one(
    const int* bin_indices,
    int n_features,
    const double* points_table,
    int max_bins,
    double base_points
) {
    double total = base_points;
    for (int f = 0; f < n_features; f++) {
        int bin = bin_indices[f];
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
    for (int row = 0; row < n_rows; row++) {
        const int* row_bins = bin_indices + (size_t)row * n_features;
        out_scores[row] = score_one(row_bins, n_features, points_table, max_bins, base_points);
    }
}
