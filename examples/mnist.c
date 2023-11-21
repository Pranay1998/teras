#include "mnist_loader.h"
#include "../teras.h"

#define OPERATOR ^

int max_index(Row r) {
    float max = -1.f;
    int max_index = -1;
    for (size_t i = 0; i < r.size; i++) {
        if (ROW_AT(r, i) > max) {
            max = ROW_AT(r, i);
            max_index = i;
        }
    }
    return max_index;
}

void evaluate(NN n, Matrix test) {
    size_t correct = 0;

    for (size_t i = 0; i < test.rows; i++) {
        Row t = mat_row(test, i);
        Row x = row_slice(t, 0, NN_INPUT(n).size); 
        Row y = row_slice(t, NN_INPUT(n).size, NN_OUTPUT(n).size);

        row_copy(NN_INPUT(n), x);
        nn_forward(n);

        Row prediction = NN_OUTPUT(n);

        if (max_index(prediction) == max_index(y)) {
            correct++;
        }
    }

    printf("Accuracy - %zu / %zu\n", correct, test.rows);
}

int main() {
    load_mnist();
    Matrix train = matrix_alloc(60000, 794);
    Matrix test = matrix_alloc(10000, 794);

    for (size_t i = 0; i < 60000; i++) {
        for (size_t j = 0; j < 794; j++) {
            if (j < 784) {
                MATRIX_AT(train, i, j) = (float) train_image[i][j];
            } else if (train_label[i] == (j - 784)) {
                MATRIX_AT(train, i, j) = 1.f;
            } else {
                MATRIX_AT(train, i, j) = 0.f;
            }
        }
    }

    for (size_t i = 0; i < 10000; i++) {
        for (size_t j = 0; j < 794; j++) {
            if (j < 784) {
                MATRIX_AT(test, i, j) = (float) test_image[i][j];
            } else if (test_label[i] == (j - 784)) {
                MATRIX_AT(test, i, j) = 1.f;
            } else {
                MATRIX_AT(test, i, j) = 0.f;
            }
        }
    }

    size_t layers[] = {784, 20, 10};
    Act activations[] = {Sigmoid, Sigmoid};
    NN n = nn_alloc(layers, activations, ARR_LEN(layers));
    nn_rand(n);

    float rate = 3;
    size_t batch_size = 10;
    size_t epochs = 200;

    nn_sgd(n, train, epochs, batch_size, rate, test, &evaluate);
}
