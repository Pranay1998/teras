#include "teras.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define matrix_at(m, row, column) m.data[(row)*m.columns+(column)]
 
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
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            printf("%f ", matrix_at(m, row, column));
        }
        printf("\n");
    }
}

void matrix_fill_rand(Matrix m) { 
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            matrix_at(m, row, column) = rand_float();
        }
    }
}

void matrix_dot(Matrix dest, Matrix a, Matrix b) {
    assert(dest.rows == a.rows && dest.columns == b.columns);
    assert(a.columns == b.rows);
    size_t n = a.columns;

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            matrix_at(dest, row, column) = 0.f;
            for (size_t inner = 0; inner < n; inner++) {
                matrix_at(dest, row, column) += matrix_at(a, row, inner) * matrix_at(b, inner, column);
            }
        }
    }
}

void matrix_sum(Matrix dest, Matrix a, Matrix b) {
    assert(a.rows == b.rows && a.columns == b.columns);
    assert(dest.rows == a.rows && dest.columns == a.columns);

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            matrix_at(dest, row, column) = matrix_at(a, row, column) + matrix_at(b, row, column);
        }
    }
}

int main(void) {
    Matrix a = matrix_create(2, 3);
    Matrix b = matrix_create(3, 2);
    matrix_fill_rand(a);
    matrix_fill_rand(b);
    
    matrix_print(a);
    printf("\n*\n\n");
    matrix_print(b);

    Matrix c = matrix_create(2, 2);

    matrix_dot(c, a, b);
    printf("\n=\n\n");
    matrix_print(c);

    return 0;
}
