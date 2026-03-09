#include "matrix.h"

#ifdef __arm__
#include "arm_math.h"
#endif

/**
 * @brief Progress callback for monitoring matrix operations.
 */
void (*matrix_row_callback)(uint16_t row) = NULL;

void matrix_multiply(const Matrix* matrix_a, const Matrix* matrix_b, Matrix* result_matrix) {
#ifdef __arm__
    // Hardware accelerated version using CMSIS-DSP
    // This utilizes Cortex-M4 SIMD and FPU instructions.
    arm_matrix_instance_f32 A = {matrix_a->rows, matrix_a->columns, matrix_a->data};
    arm_matrix_instance_f32 B = {matrix_b->rows, matrix_b->columns, matrix_b->data};
    arm_matrix_instance_f32 R = {result_matrix->rows, result_matrix->columns, result_matrix->data};

    arm_mat_mult_f32(&A, &B, &R);
#else
    // Standard C fallback for Host tests
    for (uint16_t i = 0; i < matrix_a->rows; i++) {
        for (uint16_t j = 0; j < matrix_b->columns; j++) {
            float sum = 0.0f;
            for (uint16_t k = 0; k < matrix_a->columns; k++) {
                sum += matrix_a->data[i * matrix_a->columns + k] *
                       matrix_b->data[k * matrix_b->columns + j];
            }
            result_matrix->data[i * matrix_b->columns + j] = sum;
        }
    }
#endif
}

void matrix_multiply_q15(const MatrixQ15* matrix_a, const MatrixQ15* matrix_b,
                         MatrixQ15* result_matrix) {
#ifdef __arm__
    // Hardware accelerated fixed-point version
    arm_matrix_instance_q15 A = {matrix_a->rows, matrix_a->columns, matrix_a->data};
    arm_matrix_instance_q15 B = {matrix_b->rows, matrix_b->columns, matrix_b->data};
    arm_matrix_instance_q15 R = {result_matrix->rows, result_matrix->columns, result_matrix->data};

    arm_mat_mult_q15(&A, &B, &R, NULL);
#else
    // Fixed-point C fallback for Host tests
    for (uint16_t i = 0; i < matrix_a->rows; i++) {
        for (uint16_t j = 0; j < matrix_b->columns; j++) {
            int32_t sum = 0;
            for (uint16_t k = 0; k < matrix_a->columns; k++) {
                sum += (int32_t)matrix_a->data[i * matrix_a->columns + k] *
                       matrix_b->data[k * matrix_b->columns + j];
            }
            result_matrix->data[i * matrix_b->columns + j] = (int16_t)(sum >> 15);
        }
    }
#endif
}
