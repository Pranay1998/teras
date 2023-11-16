#include "teras.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
 
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

Matrix row_as_matrix(Row row) {
    return (Matrix) {
        .rows = 1,
        .columns = row.size,
        .data = row.data
    };
}

Row row_slice(Row row, size_t start, size_t size) {
    TERAS_ASSERT(start < row.size);
    TERAS_ASSERT(start + size <= row.size);
    return (Row) {
        .size = size,
        .data = &ROW_AT(row, start)
    };
}

Matrix matrix_create(size_t rows, size_t columns) {
    TERAS_ASSERT(rows >= 1);
    TERAS_ASSERT(columns >= 1);
    
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

void matrix_copy(Matrix dest, Matrix a) {
    TERAS_ASSERT(dest.rows == a.rows);
    TERAS_ASSERT(dest.columns == a.columns);

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            MATRIX_AT(dest, row, column) = MATRIX_AT(a, row, column);
        }
    }
}

void matrix_rand(Matrix m) { 
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            MATRIX_AT(m, row, column) = rand_float();
        }
    }
}

void matrix_dot(Matrix dest, Matrix a, Matrix b) {
    TERAS_ASSERT(dest.rows == a.rows && dest.columns == b.columns);
    TERAS_ASSERT(a.columns == b.rows);
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

void matrix_sum(Matrix dest, Matrix a) {
    TERAS_ASSERT(dest.rows == a.rows && dest.columns == a.columns);

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            MATRIX_AT(dest, row, column) += MATRIX_AT(a, row, column);
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

Row mat_row(Matrix m, size_t row) {
    TERAS_ASSERT(row < m.rows);
    return (Row) {
        .size = m.columns,
        .data = &MATRIX_AT(m, row, 0)
    };
}

NN nn_create(size_t *layers, size_t num_layers) {
    NN n;

    TERAS_ASSERT(num_layers > 0);

    n.count = num_layers - 1;
    n.ws = malloc(sizeof(*n.ws)*n.count);
    TERAS_ASSERT(n.ws != NULL);
    n.bs = malloc(sizeof(*n.bs)*n.count);
    TERAS_ASSERT(n.bs != NULL);
    n.as = malloc(sizeof(*n.as)*(n.count + 1));
    TERAS_ASSERT(n.as != NULL);

    n.as[0] = row_create(layers[0]);

    for (int i = 0; i < n.count; i++) {
        n.ws[i] = matrix_create(layers[i], layers[i+1]);
        n.bs[i] = row_create(layers[i+1]);
        n.as[i+1] = row_create(layers[i+1]);
    }

    return n;
}

void nn_print(NN n) {
    // todo: printing will only work for i < 9
    for (size_t i = 0; i < n.count; i++) {
        char w[] = {'w', i+'0', '\0'};
        char b[] = {'b', i+'0', '\0'};
        char a[] = {'a', i+'1', '\0'};
        matrix_print(n.ws[i], w);
        matrix_print(row_as_matrix(n.bs[i]), b);
    }
}

void nn_rand(NN n) {
    for (size_t i = 0; i < n.count; i++) {
        matrix_rand(n.ws[i]);
        row_rand(n.bs[i]);
    }
}

void nn_forward(NN n) {
    for (int i = 0; i < n.count; i++) {
        matrix_dot(row_as_matrix(n.as[i+1]), row_as_matrix(n.as[i]), n.ws[i]);
        matrix_sum(row_as_matrix(n.as[i+1]), row_as_matrix(n.bs[i]));
        matrix_sigmoid(row_as_matrix(n.as[i+1]));
    }
}

float nn_cost(NN n, Matrix train) {
    TERAS_ASSERT(NN_INPUT(n).size + NN_OUTPUT(n).size == train.columns);
    float sum = 0;

    for (size_t row = 0; row < train.rows; row++) {
        Row t = mat_row(train, row);
        Row x = row_slice(t, 0, NN_INPUT(n).size); 
        Row y = row_slice(t, NN_INPUT(n).size, NN_OUTPUT(n).size); 

        row_copy(NN_INPUT(n), x);
        nn_forward(n);
        
        size_t q = y.size;
        for (size_t i = 0; i < q; i++) {
            float d = ROW_AT(NN_OUTPUT(n), i) - ROW_AT(y, i); 
            sum += d*d;
        }


    }
    return sum/train.rows;
}

int main() {
    srand(time(NULL));

    size_t layers[] = {2, 1, 1};
    NN n = nn_create(layers, ARR_LEN(layers));
    nn_rand(n);
    nn_print(n);

    Matrix train = matrix_create(4, 3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            size_t row = i*2 + j;
            MATRIX_AT(train, row, 0) = i;
            MATRIX_AT(train, row, 1) = j;
            MATRIX_AT(train, row, 2) = i^j;
        }
    }

    printf("Cost - %f\n", nn_cost(n, train));
}
