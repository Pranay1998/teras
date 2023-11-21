#ifndef TERAS_H_
#define TERAS_H_

#include <stddef.h>
#include <stdbool.h>

#ifndef TERAS_ASSERT
#include <assert.h>
#define TERAS_ASSERT assert
#endif // TERAS_ASSERT

#ifndef TERAS_ALLOC
#include <stdlib.h>
#define TERAS_ALLOC malloc
#endif // TERAS_ALLOC

#define ARR_LEN(m) sizeof(m)/sizeof(m[0])

typedef struct {
    size_t size;
    float *data;
} Row;

#define row_alloc(size) mat_row(matrix_alloc(1, size), 0)
#define row_free(r) matrix_free(row_as_matrix(r))
#define row_clone(r) mat_row(matrix_alloc(1, (r).size), 0)
#define row_rand(r) matrix_rand(row_as_matrix(r))
#define row_copy(dest, src) matrix_copy(row_as_matrix(dest), row_as_matrix(src))
#define ROW_AT(row, index) (row).data[index]

typedef struct {
    size_t rows;
    size_t columns;
    float *data;
} Matrix;

#define MATRIX_PRINT(m) matrix_print(m, #m) 
#define MATRIX_AT(m, row, column) m.data[(row)*m.columns+(column)]

typedef enum {
    Sigmoid,
    ReLu,
    LeakyReLu
} Act;

typedef struct {
    size_t *layers; // architecture of neural network
    size_t count; // num_layers - 1
    Row *zs; // intermediate values
    Row *as; // activation values
    Matrix *ws; // weights
    Row *bs; // biases
    Row *deltas; // errors
    Act *acts; // Activation functions
} NN;

#define NN_INPUT(n) (TERAS_ASSERT((n).count > 0), (n).as[0])
#define NN_OUTPUT(n) (TERAS_ASSERT((n).count > 0), (n).as[(n).count])
#define NN_PRINT(n) nn_print(n, #n)

Matrix row_as_matrix(Row row);
Row row_slice(Row row, size_t start, size_t end);

Matrix matrix_alloc(size_t rows, size_t columns);
void matrix_free(Matrix m);
void matrix_print(Matrix m, char *name);
void matrix_copy(Matrix dest, Matrix a);
void matrix_rand(Matrix dest);
void matrix_fill(Matrix m, float value);
void matrix_shuffle_rows(Matrix r);
void matrix_dot(Matrix dest, Matrix a, Matrix b, bool plus_equal);
void matrix_dot_a_transpose(Matrix dest, Matrix a, Matrix b, bool plus_equal);
void matrix_dot_b_transpose(Matrix dest, Matrix a, Matrix b, bool plus_equal);
void matrix_hadamard_product(Matrix dest, Matrix a, Matrix b);
void matrix_sum(Matrix dest, Matrix a, Matrix b);
void matrix_sigmoid(Matrix dest, Matrix m);
void matrix_sigmoid_prime(Matrix dest, Matrix m);
Row mat_row(Matrix m, size_t row);

NN nn_alloc(size_t *layers, Act *acts, size_t num_layers);
void nn_print(NN n, char *name);
void nn_rand(NN n);
void nn_fill(NN m, float value);
void nn_forward(NN n);
float nn_cost(NN n, Matrix train);
void nn_cost_derivative(Row dest, Row y, Row output_activations);
void nn_backprop(NN n, NN g, Row x, Row y);
void nn_learn(NN n, NN g, size_t batch_size, float rate);
void nn_sgd(NN n, Matrix train, size_t epochs, size_t batch_size, float rate, Matrix test, void (*evaluation_function) (NN, Matrix));

#endif
