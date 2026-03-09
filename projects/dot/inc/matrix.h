/**
 * @file dot_matrix.h
 * @brief part of the dot ML library.
 * High-performance matrix operations optimized for EDU-CIAA-NXP.
 */

#ifndef DOT_MATRIX_H
#define DOT_MATRIX_H

#include <stddef.h>
#include <stdint.h>

/**
 * @name Memory Bank Placement Macros
 * Leverages the LPC4337 AHB bus matrix by striping data across banks.
 * @{
 */
#define SRAM_BANK_A __attribute__((section(".data.$RAM2"))) /**< Local SRAM Bank A */
#define SRAM_BANK_B __attribute__((section(".data.$RAM3"))) /**< Local SRAM Bank B */
#define SRAM_BANK_C __attribute__((section(".data.$RAM1"))) /**< Local SRAM Bank C */
/** @} */

/**
 * @brief Floating-point matrix structure.
 */
typedef struct {
    uint16_t rows;
    uint16_t columns;
    float* data;
} Matrix;

/**
 * @brief 16-bit Fixed-point matrix structure (Q15 format).
 */
typedef struct {
    uint16_t rows;
    uint16_t columns;
    int16_t* data;
} MatrixQ15;

/**
 * @brief Performs the most performant matrix multiplication for the hardware.
 * Uses IKJ loop reordering, 4x loop unrolling, and pointer arithmetic to
 * saturate the Cortex-M4F FPU pipeline and minimize AHB bus stalls.
 */
void matrix_multiply(const Matrix* matrix_a, const Matrix* matrix_b, Matrix* result_matrix);

/**
 * @brief Performs 16-bit fixed-point (Q15) matrix multiplication.
 * Uses CMSIS-DSP arm_mat_mult_q15 on ARM or a fallback on Host.
 */
void matrix_multiply_q15(const MatrixQ15* matrix_a, const MatrixQ15* matrix_b,
                         MatrixQ15* result_matrix);

#endif  // DOT_MATRIX_H
