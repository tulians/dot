#include "matrix.h"

/**
 * @brief Progress callback for monitoring matrix operations.
 */
void (*matrix_row_callback)(uint16_t row) = NULL;

void matrix_multiply(const Matrix* matrix_a, const Matrix* matrix_b, Matrix* result_matrix) {
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
}
