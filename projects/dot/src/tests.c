#include "tests.h"

#include <math.h>

#include "matrix.h"
#include "nn.h"

#define EPSILON 0.00001f

static bool floats_equal(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

static bool test_matrix_multiplication_ikj(void) {
    // 2x2 Matrix A
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Matrix a = {2, 2, a_data};

    // 2x2 Identity Matrix B
    float b_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    Matrix b = {2, 2, b_data};

    // Result Matrix C
    float c_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Matrix c = {2, 2, c_data};

    // Run multiplication
    matrix_multiply(&a, &b, &c);

    // Validate C == A
    for (int i = 0; i < 4; i++) {
        if (!floats_equal(c.data[i], a.data[i])) return false;
    }
    return true;
}

static bool test_relu_activation(void) {
    float data[4] = {-1.0f, 0.0f, 2.0f, -0.5f};
    float expected[4] = {0.0f, 0.0f, 2.0f, 0.0f};
    Matrix m = {1, 4, data};

    activation_relu(&m);

    for (int i = 0; i < 4; i++) {
        if (!floats_equal(m.data[i], expected[i])) return false;
    }
    return true;
}

bool run_unit_tests(void) {
    if (!test_matrix_multiplication_ikj()) return false;
    if (!test_relu_activation()) return false;

    return true;
}
