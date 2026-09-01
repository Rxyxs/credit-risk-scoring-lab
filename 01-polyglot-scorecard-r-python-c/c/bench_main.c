/* Benchmark standalone en C puro del motor de scoring: genera un lote
 * sintetico de bin_indices y una tabla de puntos, y mide throughput real
 * de score_batch. Sirve como referencia de "techo" de velocidad, sin la
 * capa de ctypes/Python encima (ver benchmark.py para la comparacion
 * C vs Python/NumPy con la capa de interoperabilidad incluida). */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "score_engine.h"

#define N_FEATURES 5
#define MAX_BINS 8

int main(int argc, char** argv) {
    long n_rows = (argc > 1) ? atol(argv[1]) : 2000000L;

    int* bin_indices = (int*)malloc(sizeof(int) * (size_t)n_rows * N_FEATURES);
    double* out_scores = (double*)malloc(sizeof(double) * (size_t)n_rows);
    double points_table[N_FEATURES * MAX_BINS];

    if (!bin_indices || !out_scores) {
        fprintf(stderr, "No se pudo reservar memoria para %ld filas\n", n_rows);
        return 1;
    }

    srand(42);
    for (int i = 0; i < N_FEATURES * MAX_BINS; i++) {
        points_table[i] = (double)(rand() % 4000) / 100.0 - 20.0;
    }
    for (long row = 0; row < n_rows; row++) {
        for (int f = 0; f < N_FEATURES; f++) {
            bin_indices[row * N_FEATURES + f] = rand() % MAX_BINS;
        }
    }

    clock_t start = clock();
    score_batch(bin_indices, (int)n_rows, N_FEATURES, points_table, MAX_BINS, 563.8, out_scores);
    clock_t end = clock();

    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double rows_per_sec = elapsed_sec > 0 ? (double)n_rows / elapsed_sec : 0.0;

    double checksum = 0.0;
    for (long row = 0; row < n_rows; row++) checksum += out_scores[row];

    printf("filas=%ld tiempo=%.4fs throughput=%.0f filas/seg checksum=%.2f\n",
           n_rows, elapsed_sec, rows_per_sec, checksum);

    free(bin_indices);
    free(out_scores);
    return 0;
}
