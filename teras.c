#include "teras.h"
#include <math.h>
#include <stdio.h>
#include <time.h> 

size_t min(size_t a, size_t b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

float sigmoidf_prime(float x) {
    return sigmoidf(x) * (1 - sigmoidf(x));
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

Matrix matrix_alloc(size_t rows, size_t columns) {
    TERAS_ASSERT(rows >= 1);
    TERAS_ASSERT(columns >= 1);
    
    Matrix m;
    m.rows = rows;
    m.columns = columns;
    m.data = TERAS_MALLOC(sizeof(*m.data) * rows * columns);
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

void matrix_shuffle_rows(Matrix train) {
    float temp;
    
    int n = train.rows;

    if (n > 1) {
        for (size_t i = 0; i < n; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            for (size_t column = 0; column < train.columns; column++) {
                temp = MATRIX_AT(train, i, column);
                MATRIX_AT(train, i, column) = MATRIX_AT(train, j, column);
                MATRIX_AT(train, j, column) = temp;
            }
        }
    }
}

void matrix_dot(Matrix dest, Matrix a, Matrix b, bool plus_equal) {
    TERAS_ASSERT(dest.rows == a.rows && dest.columns == b.columns);
    TERAS_ASSERT(a.columns == b.rows);
    size_t n = a.columns; // = b.rows

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            if (!plus_equal) {
                MATRIX_AT(dest, row, column) = 0.f;
            }
            for (size_t inner = 0; inner < n; inner++) {
                MATRIX_AT(dest, row, column) += MATRIX_AT(a, row, inner) * MATRIX_AT(b, inner, column);
            }
        }
    }
}

void matrix_dot_a_transpose(Matrix dest, Matrix a, Matrix b, bool plus_equal) {
    TERAS_ASSERT(dest.rows == a.columns && dest.columns == b.columns);
    TERAS_ASSERT(a.rows == b.rows);
    size_t n = a.rows; // = b.rows

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            if (!plus_equal) {
                MATRIX_AT(dest, row, column) = 0.f;
            }
            for (size_t inner = 0; inner < n; inner++) {
                MATRIX_AT(dest, row, column) += MATRIX_AT(a, inner, row) * MATRIX_AT(b, inner, column);
            }
        }
    }
}


void matrix_dot_b_transpose(Matrix dest, Matrix a, Matrix b, bool plus_equal) {
    TERAS_ASSERT(dest.rows == a.rows && dest.columns == b.rows);
    TERAS_ASSERT(a.columns == b.columns);
    size_t n = a.columns; // = b.columns

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            if (!plus_equal) {
                MATRIX_AT(dest, row, column) = 0.f;
            }
            for (size_t inner = 0; inner < n; inner++) {
                MATRIX_AT(dest, row, column) += MATRIX_AT(a, row, inner) * MATRIX_AT(b, column, inner);
            }
        }
    }
}

void matrix_hadamard_product(Matrix dest, Matrix a, Matrix b) {
    TERAS_ASSERT(dest.rows == a.rows && dest.columns == a.columns);
    TERAS_ASSERT(a.rows == b.rows && a.columns == b.columns);

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            MATRIX_AT(dest, row, column) = MATRIX_AT(a, row, column) * MATRIX_AT(b, row, column);
        }
    }
}

void matrix_sum(Matrix dest, Matrix a, Matrix b) {
    TERAS_ASSERT(dest.rows == a.rows && dest.columns == a.columns);
    TERAS_ASSERT(a.rows == b.rows && a.columns == b.columns);

    for (size_t row = 0; row < dest.rows; row++) {
        for (size_t column = 0; column < dest.columns; column++) {
            MATRIX_AT(dest, row, column) = MATRIX_AT(a, row, column) + MATRIX_AT(b, row, column);
        }
    }
}

void matrix_sigmoid(Matrix dest, Matrix m) {
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            MATRIX_AT(dest, row, column) = sigmoidf(MATRIX_AT(m, row, column));
        }
    }
}

void matrix_sigmoid_prime(Matrix dest, Matrix m) {
    for (size_t row = 0; row < m.rows; row++) {
        for (size_t column = 0; column < m.columns; column++) {
            MATRIX_AT(dest, row, column) = sigmoidf_prime(MATRIX_AT(m, row, column));
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

NN nn_alloc(size_t *layers, size_t num_layers) {
    NN n;

    TERAS_ASSERT(num_layers > 0);

    n.count = num_layers - 1;
    n.ws = TERAS_MALLOC(sizeof(*n.ws)*n.count);
    TERAS_ASSERT(n.ws != NULL);
    n.bs = TERAS_MALLOC(sizeof(*n.bs)*n.count);
    TERAS_ASSERT(n.bs != NULL);
    n.zs = TERAS_MALLOC(sizeof(*n.zs)*n.count);
    TERAS_ASSERT(n.zs != NULL);
    n.deltas = TERAS_MALLOC(sizeof(*n.deltas)*n.count);
    TERAS_ASSERT(n.deltas != NULL);
    n.as = TERAS_MALLOC(sizeof(*n.as)*(n.count+1));
    TERAS_ASSERT(n.as != NULL);

    n.as[0] = row_alloc(layers[0]);

    for (size_t i = 0; i < n.count; i++) {
        n.ws[i] = matrix_alloc(layers[i], layers[i+1]);
        n.bs[i] = row_alloc(layers[i+1]);
        n.zs[i] = row_alloc(layers[i+1]);
        n.deltas[i] = row_alloc(layers[i+1]);
        n.as[i+1] = row_alloc(layers[i+1]);
    }

    return n;
}

void nn_print(NN n, char *name) {
    printf("------------------------------------\n");
    printf("Neural network = %s\n", name);
    for (size_t i = 0; i < n.count; i++) {
        char name_w[10];
        char name_b[10];

        sprintf(name_w, "w%zu", i);
        sprintf(name_b, "b%zu", i);

        matrix_print(n.ws[i], name_w);
        matrix_print(row_as_matrix(n.bs[i]), name_b);
    }
    printf("------------------------------------\n");
}

void nn_rand(NN n) {
    for (size_t i = 0; i < n.count; i++) {
        matrix_rand(n.ws[i]);
        row_rand(n.bs[i]);
    }
}

void nn_forward(NN n) {
    for (int i = 0; i < n.count; i++) {
        matrix_dot(row_as_matrix(n.zs[i]), row_as_matrix(n.as[i]), n.ws[i], false);
        matrix_sum(row_as_matrix(n.zs[i]), row_as_matrix(n.zs[i]), row_as_matrix(n.bs[i]));
        matrix_sigmoid(row_as_matrix(n.as[i+1]), row_as_matrix(n.zs[i]));
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
    return sum/2;
}

void nn_cost_derivative(Row dest, Row y, Row output_activations) {
    TERAS_ASSERT(dest.size == y.size);
    TERAS_ASSERT(y.size == output_activations.size);

    for (size_t i = 0; i < dest.size; i++) {
        ROW_AT(dest, i) = ROW_AT(output_activations, i) - ROW_AT(y, i);
    }
}

void nn_backprop(NN n, NN g, Row x, Row y) {
    TERAS_ASSERT(x.size == NN_INPUT(n).size);
    TERAS_ASSERT(y.size == NN_OUTPUT(n).size);

    size_t num_layers = n.count;
    row_copy(NN_INPUT(n), x);
    nn_forward(n);

    Row delta = n.deltas[num_layers-1];
    nn_cost_derivative(g.zs[num_layers-1], y, NN_OUTPUT(n)); // temporary store
    matrix_sigmoid_prime(row_as_matrix(delta), row_as_matrix(n.zs[num_layers-1]));
    matrix_hadamard_product(row_as_matrix(delta), row_as_matrix(delta), row_as_matrix(g.zs[num_layers-1]));

    row_copy(g.bs[num_layers-1], delta);
    matrix_dot_a_transpose(g.ws[num_layers-1], row_as_matrix(n.as[num_layers-1]), row_as_matrix(delta), false);

    for (size_t l = num_layers-2; l != (size_t) -1; l--) {
        Row delta_l = n.deltas[l];
        matrix_sigmoid_prime(row_as_matrix(g.zs[l]), row_as_matrix(n.zs[l])); // temporary store

        matrix_dot_b_transpose(row_as_matrix(delta_l), row_as_matrix(delta), n.ws[l+1], false);
        matrix_hadamard_product(row_as_matrix(delta_l), row_as_matrix(delta_l), row_as_matrix(g.zs[l]));
        delta = delta_l; 

        row_copy(g.bs[l], delta);
        matrix_dot_a_transpose(g.ws[l], row_as_matrix(n.as[l]), row_as_matrix(delta), false);
    }
}

void nn_learn(NN n, NN *gs, size_t batch_size, float rate) { 
    for (size_t batch = 0; batch < batch_size; batch++) {
        NN g = gs[batch];
        for (size_t i = 0; i < n.count; i++) {
            assert(n.ws[i].rows == g.ws[i].rows);
            assert(n.ws[i].columns == g.ws[i].columns);
            assert(n.bs[i].size == g.bs[i].size);

            for (size_t row = 0; row < n.ws[i].rows; row++) {
                for (size_t column = 0; column < n.ws[i].columns; column++) {
                    MATRIX_AT(n.ws[i], row, column) -= (rate*MATRIX_AT(g.ws[i], row, column))/batch_size;
                }
            }

            for (size_t size = 0; size < n.bs[i].size; size++) {
                ROW_AT(n.bs[i], size) -= (rate*ROW_AT(g.bs[i], size))/batch_size;
            }
        }
    }
}

void nn_sgd(NN n, NN *g, Matrix train, size_t epochs, size_t batch_size, float rate) {
    TERAS_ASSERT(batch_size <= train.rows);
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        
        if (batch_size < train.rows) {
            matrix_shuffle_rows(train);
        }

        size_t batch_start = 0;
        while (batch_start < train.rows) {
            size_t batch_end = min(batch_start + batch_size, train.rows - 1);

            TERAS_ASSERT(batch_start + batch_end - 1 <= batch_size);

            for (size_t i = batch_start; i <= batch_end; i++) {
                Row t = mat_row(train, i);
                Row x = row_slice(t, 0, NN_INPUT(n).size); 
                Row y = row_slice(t, NN_INPUT(n).size, NN_OUTPUT(n).size);
                nn_backprop(n, g[i-batch_start], x, y);
            }
            nn_learn(n, g, batch_end - batch_start + 1, rate);

            batch_start = batch_end + 1;
        }
    }
}
