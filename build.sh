#!/bin/bash

clang -o3 examples/gates.c teras.c -o gates -lm

clang -o3 tests.c teras.c -o tests -lm

clang -o3 examples/mnist.c teras.c -o mnist -lm