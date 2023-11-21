#include <stdio.h>
#include <string.h>
#include "../teras.h"
#include "time.h"

#define OPERATOR ^

void evaluate(NN n, Matrix test) {

}

int main() {
    srand(time(NULL));
    size_t layers[] = {2, 2, 1};
    Act acts[] = {Sigmoid, Sigmoid};
    NN n = nn_alloc(layers, acts, ARR_LEN(layers));
    nn_rand(n);

    float dataset[] = {
        0, 0, 0 OPERATOR 0,
        0, 1, 0 OPERATOR 1,
        1, 0, 1 OPERATOR 0,
        1, 1, 1 OPERATOR 1
    };
    Matrix train = matrix_alloc(4, 3);
    memcpy(train.data, dataset, sizeof(dataset));

    float rate = 10;
    size_t batch_size = 2;
    size_t epochs = 1000;

    nn_sgd(n, train, epochs, batch_size, rate, train, evaluate);
}
