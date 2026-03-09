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
 * @brief Performs the most performant matrix multiplication for the hardware.
 * Uses IKJ loop reordering, 4x loop unrolling, and pointer arithmetic to
 * saturate the Cortex-M4F FPU pipeline and minimize AHB bus stalls.
 */
void matrix_multiply(const Matrix* matrix_a, const Matrix* matrix_b, Matrix* result_matrix);

#endif  // DOT_MATRIX_H
