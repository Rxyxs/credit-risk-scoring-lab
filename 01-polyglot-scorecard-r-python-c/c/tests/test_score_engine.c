/* Suite de pruebas unitarias de score_engine.c: corrección numérica del
 * cálculo de puntos, manejo seguro de punteros NULL y de bins fuera de
 * rango, y consistencia entre llamadas repetidas (score_one no mantiene
 * ningún estado global mutable que una corrida pueda contaminar para la
 * siguiente -- no hay memoria propia que reservar ni liberar dentro del
 * motor, así que no hay fuga posible del lado de score_engine.c mismo).
 *
 * Runner liviano con aserciones estándar de C: imprime cada fallo y
 * termina con exit code 0 si todo pasó, 1 si algo falló -- pensado para
 * invocarse desde `make test` en CI.
 */

#include <math.h>
#include <stdio.h>

#include "../score_engine.h"

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        tests_run++;                                                         \
        if (!(cond)) {                                                       \
            tests_failed++;                                                  \
            fprintf(stderr, "FALLO [linea %d]: %s -- %s\n", __LINE__, msg,   \
                    #cond);                                                  \
        }                                                                    \
    } while (0)

#define CHECK_DOUBLE_EQ(a, b, msg) CHECK(fabs((a) - (b)) <= 1e-9, msg)

/* -------------------------------------------------- score_one: correctitud */

static void test_score_one_calcula_el_valor_exacto(void) {
    /* 3 features, 4 bins cada una. bin_indices = [2, 0, 3]. */
    double points_table[12] = {
        10, 20, 30, 40,   /* feature 0 */
        -5, -10, -15, -20, /* feature 1 */
        1, 2, 3, 4         /* feature 2 */
    };
    int bin_indices[3] = {2, 0, 3};
    double base_points = 500.0;

    double score = score_one(bin_indices, 3, points_table, 4, base_points);
    /* 500 + points_table[0*4+2] + points_table[1*4+0] + points_table[2*4+3]
     *   = 500 + 30 + (-5) + 4 = 529 */
    CHECK_DOUBLE_EQ(score, 529.0, "score_one no calculo el puntaje esperado");
}

static void test_score_one_con_cero_features_devuelve_base_points(void) {
    /* n_features == 0 es un caso valido (aplicante sin features
     * seleccionadas), no un error: no deberia leer bin_indices ni
     * points_table, y el resultado es exactamente base_points. */
    double score = score_one(NULL, 0, NULL, 4, 777.25);
    CHECK_DOUBLE_EQ(score, 777.25,
                    "n_features=0 deberia devolver base_points sin tocar punteros NULL");
}

/* -------------------------------------------------- score_one: entradas invalidas */

static void test_score_one_bin_indices_null_devuelve_nan(void) {
    double points_table[4] = {1, 2, 3, 4};
    double score = score_one(NULL, 2, points_table, 4, 100.0);
    CHECK(isnan(score), "bin_indices NULL con n_features>0 deberia devolver NAN");
}

static void test_score_one_points_table_null_devuelve_nan(void) {
    int bin_indices[2] = {0, 1};
    double score = score_one(bin_indices, 2, NULL, 4, 100.0);
    CHECK(isnan(score), "points_table NULL con n_features>0 deberia devolver NAN");
}

static void test_score_one_bin_negativo_devuelve_nan(void) {
    double points_table[4] = {1, 2, 3, 4};
    int bin_indices[1] = {-1};
    double score = score_one(bin_indices, 1, points_table, 4, 100.0);
    CHECK(isnan(score), "un bin negativo deberia devolver NAN, no leer fuera de rango");
}

static void test_score_one_bin_fuera_de_max_bins_devuelve_nan(void) {
    double points_table[4] = {1, 2, 3, 4};
    int bin_indices[1] = {4}; /* max_bins=4 -> indices validos son 0..3 */
    double score = score_one(bin_indices, 1, points_table, 4, 100.0);
    CHECK(isnan(score), "un bin == max_bins deberia devolver NAN (fuera de rango), no leer memoria ajena");
}

static void test_score_one_max_bins_no_positivo_devuelve_nan(void) {
    double points_table[1] = {1};
    int bin_indices[1] = {0};
    double score = score_one(bin_indices, 1, points_table, 0, 100.0);
    CHECK(isnan(score), "max_bins <= 0 (tabla de puntos vacia) deberia devolver NAN");
}

static void test_score_one_n_features_negativo_devuelve_nan(void) {
    double score = score_one(NULL, -1, NULL, 4, 100.0);
    CHECK(isnan(score), "n_features negativo es una entrada invalida, deberia devolver NAN");
}

/* -------------------------------------------------- score_batch: entradas invalidas */

static void test_score_batch_out_scores_null_no_crashea(void) {
    int bin_indices[2] = {0, 1};
    double points_table[4] = {1, 2, 3, 4};
    /* No debe haber segfault: simplemente no hay donde escribir. */
    score_batch(bin_indices, 1, 2, points_table, 4, 0.0, NULL);
    CHECK(1, "score_batch con out_scores NULL no deberia crashear");
}

static void test_score_batch_n_rows_no_positivo_no_toca_el_buffer(void) {
    double centinela = -12345.0;
    double out_scores[1] = {-12345.0};
    int bin_indices[2] = {0, 1};
    double points_table[4] = {1, 2, 3, 4};

    score_batch(bin_indices, 0, 2, points_table, 4, 0.0, out_scores);
    CHECK_DOUBLE_EQ(out_scores[0], centinela,
                    "n_rows=0 no deberia escribir nada en out_scores");

    score_batch(bin_indices, -3, 2, points_table, 4, 0.0, out_scores);
    CHECK_DOUBLE_EQ(out_scores[0], centinela,
                    "n_rows negativo no deberia escribir nada en out_scores");
}

static void test_score_batch_punteros_null_llena_nan_sin_crashear(void) {
    double out_scores[3] = {0, 0, 0};
    double points_table[4] = {1, 2, 3, 4};

    score_batch(NULL, 3, 2, points_table, 4, 100.0, out_scores);
    CHECK(isnan(out_scores[0]) && isnan(out_scores[1]) && isnan(out_scores[2]),
          "bin_indices NULL en score_batch deberia llenar out_scores de NAN, no crashear");
}

/* ------------------------------------------- score_batch vs score_one: coherencia */

static void test_score_batch_coincide_con_llamadas_repetidas_a_score_one(void) {
    const int n_rows = 5;
    const int n_features = 3;
    const int max_bins = 4;
    double points_table[12] = {
        10, 20, 30, 40,
        -5, -10, -15, -20,
        1, 2, 3, 4
    };
    int bin_indices[15] = {
        0, 1, 2,
        3, 0, 1,
        2, 3, 0,
        1, 2, 3,
        0, 0, 0
    };
    double base_points = 563.8;
    double out_scores[5];

    score_batch(bin_indices, n_rows, n_features, points_table, max_bins, base_points, out_scores);

    for (int row = 0; row < n_rows; row++) {
        double esperado = score_one(bin_indices + row * n_features, n_features,
                                     points_table, max_bins, base_points);
        CHECK_DOUBLE_EQ(out_scores[row], esperado,
                        "score_batch deberia coincidir exactamente con score_one fila a fila");
    }
}

/* ------------------------------------------- consistencia entre corridas repetidas */

static void test_llamadas_repetidas_dan_el_mismo_resultado(void) {
    /* score_one/score_batch son funciones puras: sin estado global, sin
     * memoria propia. Si esto alguna vez deja de ser bit-identico entre
     * corridas con la misma entrada, algo introdujo estado mutable
     * indeseado (ej. una variable static). */
    double points_table[12] = {
        10, 20, 30, 40,
        -5, -10, -15, -20,
        1, 2, 3, 4
    };
    int bin_indices[3] = {2, 0, 3};

    double primero = score_one(bin_indices, 3, points_table, 4, 500.0);
    for (int i = 0; i < 1000; i++) {
        double repetido = score_one(bin_indices, 3, points_table, 4, 500.0);
        CHECK_DOUBLE_EQ(repetido, primero,
                        "score_one deberia dar el mismo resultado en 1000 llamadas repetidas");
    }

    double out_a[3];
    double out_b[3];
    int batch_bins[9] = {0, 1, 2, 3, 0, 1, 2, 3, 0};
    score_batch(batch_bins, 3, 3, points_table, 4, 500.0, out_a);
    score_batch(batch_bins, 3, 3, points_table, 4, 500.0, out_b);
    for (int i = 0; i < 3; i++) {
        CHECK_DOUBLE_EQ(out_a[i], out_b[i],
                        "score_batch deberia dar el mismo resultado en dos corridas identicas");
    }
}

int main(void) {
    test_score_one_calcula_el_valor_exacto();
    test_score_one_con_cero_features_devuelve_base_points();
    test_score_one_bin_indices_null_devuelve_nan();
    test_score_one_points_table_null_devuelve_nan();
    test_score_one_bin_negativo_devuelve_nan();
    test_score_one_bin_fuera_de_max_bins_devuelve_nan();
    test_score_one_max_bins_no_positivo_devuelve_nan();
    test_score_one_n_features_negativo_devuelve_nan();
    test_score_batch_out_scores_null_no_crashea();
    test_score_batch_n_rows_no_positivo_no_toca_el_buffer();
    test_score_batch_punteros_null_llena_nan_sin_crashear();
    test_score_batch_coincide_con_llamadas_repetidas_a_score_one();
    test_llamadas_repetidas_dan_el_mismo_resultado();

    printf("%d/%d pruebas OK\n", tests_run - tests_failed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
