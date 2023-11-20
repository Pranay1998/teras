#!/bin/bash

clang examples/gates.c teras.c -o gates -lm

clang tests.c teras.c -o tests -lm