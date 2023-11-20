#include <stdio.h>
#include "../teras.h"
#include "time.h"

#define OPERATOR ^

void evaluate(NN n, Matrix test) {

}

int main() {
    srand(time(NULL));
    size_t layers[] = {2, 2, 1};
    NN n = nn_alloc(layers, ARR_LEN(layers));
    NN g = nn_alloc(layers, ARR_LEN(layers));
    nn_rand(n);

    Matrix train = matrix_alloc(4, 3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            size_t row = i*2 + j;
            MATRIX_AT(train, row, 0) = i;
            MATRIX_AT(train, row, 1) = j;
            MATRIX_AT(train, row, 2) = i OPERATOR j;
        }
    }

    float rate = 10;
    size_t batch_size = 2;
    size_t epochs = 1000;

    nn_sgd(n, g, train, epochs, batch_size, rate, train, evaluate);
    printf("Final cost - %f\n", nn_cost(n, train));
}
