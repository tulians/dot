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
    // 1. Wrap custom Matrix into CMSIS-DSP arm_matrix_instance_f32
    arm_matrix_instance_f32 A = {matrix_a->rows, matrix_a->columns, matrix_a->data};
    arm_matrix_instance_f32 B = {matrix_b->rows, matrix_b->columns, matrix_b->data};
    arm_matrix_instance_f32 R = {result_matrix->rows, result_matrix->columns, result_matrix->data};

    // 2. Execute Hardware Accelerated Matrix Multiplication
    // This utilizes Cortex-M4 SIMD and FPU instructions.
    arm_status status = arm_mat_mult_f32(&A, &B, &R);

    // Optional: handle status error if needed
    (void)status;
#else
    // 1. Zero out result matrix using a tight loop (compiler will likely use VST)
    uint32_t total_elements = (uint32_t)result_matrix->rows * result_matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        result_matrix->data[element_index] = 0.0f;
    }

    uint16_t columns_b = matrix_b->columns;
    uint16_t columns_a = matrix_a->columns;

    // 2. Perform Matrix Multiplication using IKJ + 4x Unrolling
    for (uint16_t row_index = 0; row_index < matrix_a->rows; row_index++) {
        if (matrix_row_callback != NULL) {
            matrix_row_callback(row_index);
        }

        for (uint16_t inner_index = 0; inner_index < columns_a; inner_index++) {
            /*
             * Cache value from A to saturate FPU registers and avoid RAM fetches
             * in the innermost loop.
             */
            float value_from_a = matrix_a->data[row_index * columns_a + inner_index];

            /* Pointer arithmetic for linear memory traversal */
            float* row_b_ptr = &matrix_b->data[inner_index * columns_b];
            float* row_res_ptr = &result_matrix->data[row_index * columns_b];

            uint16_t column_index = 0;

            /*
             * 4x Unrolled Core: Minimizes branches and allows overlapping
             * floating-point operations.
             */
            for (; column_index <= (columns_b - 4); column_index += 4) {
                row_res_ptr[column_index + 0] += value_from_a * row_b_ptr[column_index + 0];
                row_res_ptr[column_index + 1] += value_from_a * row_b_ptr[column_index + 1];
                row_res_ptr[column_index + 2] += value_from_a * row_b_ptr[column_index + 2];
                row_res_ptr[column_index + 3] += value_from_a * row_b_ptr[column_index + 3];
            }

            /* Clean up loop for remaining elements */
            for (; column_index < columns_b; column_index++) {
                row_res_ptr[column_index] += value_from_a * row_b_ptr[column_index];
            }
        }
    }
#endif
}

void matrix_multiply_q15(const MatrixQ15* matrix_a, const MatrixQ15* matrix_b,
                         MatrixQ15* result_matrix) {
#ifdef __arm__
    arm_matrix_instance_q15 A = {matrix_a->rows, matrix_a->columns, matrix_a->data};
    arm_matrix_instance_q15 B = {matrix_b->rows, matrix_b->columns, matrix_b->data};
    arm_matrix_instance_q15 R = {result_matrix->rows, result_matrix->columns, result_matrix->data};

    // Use CMSIS-DSP Q15 matrix multiplication
    arm_mat_mult_q15(&A, &B, &R, NULL);
#else
    // Fallback: Simple Q15 multiplication for Host
    for (uint16_t i = 0; i < matrix_a->rows; i++) {
        for (uint16_t j = 0; j < matrix_b->columns; j++) {
            int32_t sum = 0;
            for (uint16_t k = 0; k < matrix_a->columns; k++) {
                // Fixed-point multiply and accumulate (Q15 * Q15 -> Q30)
                sum += (int32_t)matrix_a->data[i * matrix_a->columns + k] *
                       matrix_b->data[k * matrix_b->columns + j];
            }
            // Scale back to Q15 (right shift by 15)
            result_matrix->data[i * matrix_b->columns + j] = (int16_t)(sum >> 15);
        }
    }
#endif
}
