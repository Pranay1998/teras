#include "teras.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

 
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
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

void matrix_print(Matrix m, char *name) {
    printf("%s =\n", name);
    for (size_t row = 0; row < m.rows; row++) {
        printf("    |");
        for (size_t column = 0; column < m.columns; column++) {
            printf("   %f", MATRIX_AT(m, row, column));
        }
        printf("    |\n");
    }

}

void matrix_fill_rand(Matrix m) { 
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            MATRIX_AT(m, row, column) = rand_float();
        }
    }
}

void matrix_dot(Matrix dest, Matrix a, Matrix b) {
    assert(dest.rows == a.rows && dest.columns == b.columns);
    assert(a.columns == b.rows);
    size_t n = a.columns;

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            MATRIX_AT(dest, row, column) = 0.f;
            for (size_t inner = 0; inner < n; inner++) {
                MATRIX_AT(dest, row, column) += MATRIX_AT(a, row, inner) * MATRIX_AT(b, inner, column);
            }
        }
    }
}

void matrix_sum(Matrix dest, Matrix a, Matrix b) {
    assert(a.rows == b.rows && a.columns == b.columns);
    assert(dest.rows == a.rows && dest.columns == a.columns);

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            MATRIX_AT(dest, row, column) = MATRIX_AT(a, row, column) + MATRIX_AT(b, row, column);
        }
    }
}

void matrix_sigmoid(Matrix m) {
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            MATRIX_AT(m, row, column)= sigmoidf(MATRIX_AT(m, row, column));
        }
    }
}

float train_or[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1}
};

#define SIZE_OR 4


typedef struct {
    Matrix x; // inputs
    Matrix a0, w, b; // intermediete layer
    Matrix y; // output
} Or;

void forward_or(Or or, float x1, float x2) {
    MATRIX_AT(or.x, 0, 0) = x1;
    MATRIX_AT(or.x, 0, 1) = x2;
    matrix_dot(or.a0, or.x, or.w);
    matrix_sum(or.y, or.a0, or.b);
    matrix_sigmoid(or.y);
}

float cost_or(Or or) {
    float sum = 0.f;
    for (size_t i = 0; i < SIZE_OR; i++) {
        float x1 = train_or[i][0];
        float x2 = train_or[i][1];
        forward_or(or, x1, x2);
        float diff = train_or[i][2] - *or.y.data;
        sum += diff * diff;
    }
    return sum / SIZE_OR;
}

int main(void) {
    Or or;

    or.x = matrix_create(1, 2);

    or.a0 = matrix_create(1, 1);
    or.w = matrix_create(2, 1);
    or.b = matrix_create(1, 1);

    or.y = matrix_create(1, 1);

    matrix_fill_rand(or.w);
    matrix_fill_rand(or.b);

    float eps = 1e-1;
    float rate = 1e-1;

    for (size_t i = 0; i < 100*1000; i++) {
        float c = cost_or(or);
        float saved = 0.f;

        saved = MATRIX_AT(or.w, 0, 0);
        MATRIX_AT(or.w, 0, 0) += eps;
        float dw1 = (cost_or(or) - c) / eps;
        MATRIX_AT(or.w, 0, 0) = saved;
        
        saved = MATRIX_AT(or.w, 1, 0);
        MATRIX_AT(or.w, 1, 0) += eps;
        float dw2 = (cost_or(or) - c) / eps;
        MATRIX_AT(or.w, 1, 0) = saved;

        saved = MATRIX_AT(or.b, 0, 0);
        MATRIX_AT(or.b, 0, 0) += eps;
        float db = (cost_or(or) - c) / eps;
        MATRIX_AT(or.b, 0, 0) = saved;

        MATRIX_AT(or.w, 0, 0) -= dw1;
        MATRIX_AT(or.w, 1, 0) -= dw2;
        MATRIX_AT(or.b, 0, 0) -= db;

        printf("Cost - %f\n",  c);
    }

    for (size_t  i = 0; i < SIZE_OR; i++) {
        float x1 = train_or[i][0];
        float x2 = train_or[i][1];
        forward_or(or, x1, x2);
        printf("%f | %f -> %f\n", x1, x2, *or.y.data);
    }

    return 0;
}
