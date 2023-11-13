#ifndef TERAS_HEADER
#define TERAS_HEADER

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t columns;
    float *data;
} Matrix;

float rand_float();
Matrix matrix_create(size_t rows, size_t columns);
void matrix_free(Matrix m);
void matrix_print(Matrix m);
void matrix_fill_rand(Matrix dest);
void matrix_dot(Matrix dest, Matrix a, Matrix b);
void matrix_sum(Matrix dest, Matrix a, Matrix b);

#endif
