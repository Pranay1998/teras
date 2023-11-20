#include "teras.h"
#include <string.h>
#include <stdio.h>

#define CHECK(r) check(r, #r)

void check(int result, char* name) {
    if (result != 0) {
        fprintf(stderr, "Test failed - %s\n", name);
    }
}

void test_matrix_dot() {
    const float array1[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    const float array2[] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };

    const float array3[] = {
        2, 1, 1, 1,
        1, 2, 1, 1,
        1, 1, 2, 1,
        1, 1, 1, 2
    };

    const float array4[] = {
        2, 1, 1, 
        1, 1, 5, 
        1, 9, 1,
        11, 2, 1,
    };

    const float array5[] = {
        2, 1, 1, 1,
        1, 6, 4, 1,
        3, 1, 2, 1,
    };

    const float array6[] = {
        8, 9, 8, 4,
        18, 12, 15, 7,
        14, 56, 39, 11,
        27, 24, 21, 14
    };

    const float array7[] = {
        9, 9, 8, 4,
        18, 13, 15, 7,
        14, 56, 40, 11,
        27, 24, 21, 15
    };

    Matrix identity_matrix = matrix_alloc(4, 4);
    Matrix result = matrix_alloc(4, 4);
    Matrix a = matrix_alloc(4, 3);
    Matrix b = matrix_alloc(3, 4);
    Matrix c = matrix_alloc(4, 4);
    
    // Dot Product Identity
    memcpy(identity_matrix.data, array1, sizeof(array1));
    matrix_dot(result, identity_matrix, identity_matrix, false);
    CHECK(memcmp(array1, result.data, sizeof(array1)));
    CHECK(memcmp(array1, identity_matrix.data, sizeof(array1)));

    // Dot Product Identity
    memcpy(identity_matrix.data, array1, sizeof(array1));
    memcpy(result.data, array2, sizeof(array2));
    matrix_dot(result, identity_matrix, identity_matrix, false);
    CHECK(memcmp(array1, result.data, sizeof(array1)));
    CHECK(memcmp(array1, identity_matrix.data, sizeof(array1)));

    // Dot Product w/ +=
    memcpy(identity_matrix.data, array1, sizeof(array1));
    memcpy(result.data, array2, sizeof(array2));
    matrix_dot(result, identity_matrix, identity_matrix, true);
    CHECK(memcmp(array3, result.data, sizeof(array3)));
    CHECK(memcmp(array1, identity_matrix.data, sizeof(array1)));

    // c = a . b
    memcpy(a.data, array4, sizeof(array4));
    memcpy(b.data, array5, sizeof(array5));
    matrix_dot(c, a, b, false);
    CHECK(memcmp(array4, a.data, sizeof(array4)));
    CHECK(memcmp(array5, b.data, sizeof(array5)));   
    CHECK(memcmp(array6, c.data, sizeof(array6)));

    // c += a . b
    memcpy(a.data, array4, sizeof(array4));
    memcpy(b.data, array5, sizeof(array5));
    memcpy(c.data, array1, sizeof(array1));
    matrix_dot(c, a, b, true);
    CHECK(memcmp(array4, a.data, sizeof(array4)));
    CHECK(memcmp(array5, b.data, sizeof(array5)));   
    CHECK(memcmp(array7, c.data, sizeof(array7)));

    matrix_free(identity_matrix);
    matrix_free(result);
    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

void test_matrix_dot_transpose() {
    const float array1[] = {
        1, 2, 3,
        2, 3, 1,
        3, 1, 2,
        1, 2, 3,
    };

    const float array2[] = {
        1, 2, 3,
        2, 3, 1,
        3, 1, 2,
        1, 2, 3,
    };

    const float array3[] = {
        15, 13, 14,
        13, 18, 17,
        14, 17, 23
    };

    const float array4[] = {
        14, 11, 11, 14,
        11, 14, 11, 11,
        11, 11, 14, 11,
        14, 11, 11, 14
    };

    const float array5[] = {
        1, 0, 2, 3,
        0, 2, 4, 6,
        1, 1, 1, 9,
        2, 0, 8, 2
    };

    const float array6[] = {
        1, 4, 1, 0,
        0, 5, 0, 1,
        1, 2, 3, 4,
        5, 1, 4, 2
    };

    const float array7[] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };

    const float array8[] = {
        13, 9, 13, 9,
        2, 13, 4, 7,
        44, 39, 38, 25,
        23, 63, 39, 47
    };

    const float array9[] = {
        16, 12, 32, 28,
        14, 29, 44, 37,
        50, 53, 80, 53,
        33, 65, 73, 93
    };

    Matrix a = matrix_alloc(4, 3);
    Matrix b = matrix_alloc(4, 3);
    Matrix c = matrix_alloc(3, 3);
    Matrix d = matrix_alloc(4, 4);
    Matrix e = matrix_alloc(4 ,4);
    Matrix f = matrix_alloc(4, 4);

    memcpy(a.data, array1, sizeof(array1));
    memcpy(b.data, array2, sizeof(array2));
    matrix_dot_a_transpose(c, a, b, false);
    matrix_dot_b_transpose(d, a, b, false);

    CHECK(memcmp(c.data, array3, sizeof(array3)));
    CHECK(memcmp(d.data, array4, sizeof(array4)));

    memcpy(d.data, array5, sizeof(array5));
    memcpy(e.data, array6, sizeof(array6));
    memcpy(f.data, array7, sizeof(array7));
    
    matrix_dot_a_transpose(f, d, e, true);
    CHECK(memcmp(f.data, array8, sizeof(array8)));

    matrix_dot_b_transpose(f, d, e, true);
    CHECK(memcmp(f.data, array9, sizeof(array9)));

    CHECK(memcmp(d.data, array5, sizeof(array5)));
    CHECK(memcmp(e.data, array6, sizeof(array6)));

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
    matrix_free(d);
    matrix_free(e);
    matrix_free(f);   
}

void test_hadamard_product() {
    const float array1[] = {
        1, 2, 30,
        2, 5, 1,
        30, 1, 2,
        1, 2, 2,
    };

    const float array2[] = {
        1, 2, 3,
        2, 3, 1,
        3, 1, 2,
        1, 2, 3,
    };

    const float array3[] = {
        1, 4, 90,
        4, 15, 1,
        90, 1, 4,
        1, 4, 6,
    };

    Matrix a = matrix_alloc(4, 3);
    Matrix b = matrix_alloc(4, 3);
    Matrix c = matrix_alloc(4, 3);

    memcpy(a.data, array1, sizeof(array1));
    memcpy(b.data, array2, sizeof(array2));
    matrix_hadamard_product(c, a, b);
    CHECK(memcmp(c.data, array3, sizeof(array3)));

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

int main() {
    test_matrix_dot();
    test_matrix_dot_transpose();
    test_hadamard_product();
    return 0;
}