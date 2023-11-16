#ifndef TERAS_H_
#define TERAS_H_

#include <stddef.h>

#ifndef TERAS_ASSERT
#include <assert.h>
#define TERAS_ASSERT assert
#endif // TERAS_ASSERT

#define ARR_LEN(m) sizeof(m)/sizeof(m[0])

float rand_float();
float sigmoidf(float x);

typedef struct {
    size_t size;
    float *data;
} Row;

#define row_create(size) mat_row(matrix_create(1, size), 0)
#define ROW_AT(row, index) (row).data[index]
#define row_rand(r) matrix_rand(row_as_matrix(r))
#define row_copy(dest, src) matrix_copy(row_as_matrix(dest), row_as_matrix(src))

typedef struct {
    size_t rows;
    size_t columns;
    float *data;
} Matrix;

#define MATRIX_PRINT(m) matrix_print(m, #m) 
#define MATRIX_AT(m, row, column) m.data[(row)*m.columns+(column)]

typedef struct {
    size_t count;
    Row *as;
    Matrix *ws;
    Row *bs;
} NN;

#define NN_INPUT(n) (TERAS_ASSERT((n).count > 0), (n).as[0])
#define NN_OUTPUT(n) (TERAS_ASSERT((n).count > 0), (n).as[(n).count-1])

Matrix row_as_matrix(Row row);
Row row_slice(Row row, size_t start, size_t end);

Matrix matrix_create(size_t rows, size_t columns);
void matrix_free(Matrix m);
void matrix_print(Matrix m, char *name);
void matrix_copy(Matrix dest, Matrix a);
void matrix_rand(Matrix dest);
void matrix_dot(Matrix dest, Matrix a, Matrix b);
void matrix_sum(Matrix dest, Matrix a);
void matrix_sigmoid(Matrix m);
Row mat_row(Matrix m, size_t row);

NN nn_create(size_t *layers, size_t num_layers);
void nn_free(NN n);
void nn_print(NN n);
void nn_rand(NN n);
void nn_forward(NN n);
float nn_cost(NN n, Matrix train);

#endif
