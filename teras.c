#include "teras.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define matrix_at(m, i, j) m.data[(i)*m.columns+(j)]
 
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

Matrix matrix_create(size_t rows, size_t columns) {
    assert(rows >= 1);
    assert(columns >= 1);
    
    Matrix m;
    m.rows = rows;
    m.columns = columns;
    m.data = malloc(sizeof(*m.data) * rows * columns);
    return m;
}

void matrix_free(Matrix m) {
    free(m.data);
}

void matrix_print(Matrix m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.columns; j++) {
            printf("%f ", matrix_at(m, i, j));
        }
        printf("\n");
    }
}

void matrix_fill_rand(Matrix m) { 
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.columns; j++) {
            matrix_at(m, i, j) = rand_float();
        }
    }
}

void matrix_dot(Matrix dest, Matrix a, Matrix b) {
    assert(a.columns == b.rows);
    assert(dest.rows == a.rows && dest.columns == b.columns);
}

void matrix_sum(Matrix dest, Matrix a, Matrix b);

int main(void) {
    Matrix test = matrix_create(0, 9);
    matrix_fill_rand(test);
    matrix_print(test);
    matrix_free(test);
    return 0;
}
