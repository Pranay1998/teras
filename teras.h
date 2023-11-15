#ifndef TERAS_HEADER
#define TERAS_HEADER

#include <stddef.h>

#define MATRIX_PRINT(m) matrix_print(m, #m) 
#define MATRIX_AT(m, row, column) m.data[(row)*m.columns+(column)]

typedef struct {
    size_t rows;
    size_t columns;
    float *data;
} Matrix;

float rand_float();
float sigmoidf(float x);

Matrix matrix_create(size_t rows, size_t columns);
void matrix_free(Matrix m);
void matrix_print(Matrix m, char *name);
void matrix_fill_rand(Matrix dest);
void matrix_dot(Matrix dest, Matrix a, Matrix b);
void matrix_sum(Matrix dest, Matrix a, Matrix b);
void matrix_sigmoid(Matrix m);

#endif
