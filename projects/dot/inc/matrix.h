/**
 * @file matrix.h
 * @brief Optimized matrix operations for embedded ML.
 *
 * DESIGN DECISIONS:
 * 1. Struct-based: Pairs metadata (rows/cols) with data for safety.
 * 2. Static-first: Macros enforce compile-time allocation to prevent heap fragmentation.
 * 3. Bank-aware: SRAM_BANK macros allow striping data across LPC4337 memory banks to
 *    maximize AHB bus throughput during heavy math.
 */

#ifndef DOT_MATRIX_H
#define DOT_MATRIX_H

#include <stddef.h>
#include <stdint.h>

/* Memory bank placement for LPC4337 multi-layer bus matrix optimization */
#define SRAM_BANK_A __attribute__((section(".data.$RAM2")))
#define SRAM_BANK_B __attribute__((section(".data.$RAM3")))
#define SRAM_BANK_C __attribute__((section(".data.$RAM1")))

typedef struct {
    uint16_t rows;
    uint16_t columns;
    float* data;
} Matrix;

typedef struct {
    uint16_t rows;
    uint16_t columns;
    int16_t* data;
} MatrixQ15;

/* Enforces static allocation to ensure deterministic memory usage */
#define DEFINE_MATRIX(name, r, c, bank) \
    float name##_data[(r) * (c)] bank;  \
    Matrix name = {r, c, name##_data}

#define DEFINE_MATRIX_Q15(name, r, c, bank) \
    int16_t name##_data[(r) * (c)] bank;    \
    MatrixQ15 name = {r, c, name##_data}

/**
 * @brief F32 Matrix Multiplication.
 * Uses CMSIS-DSP on ARM for hardware FPU/SIMD acceleration.
 */
void matrix_multiply(const Matrix* matrix_a, const Matrix* matrix_b, Matrix* result_matrix);

/**
 * @brief Q15 (Fixed-point) Matrix Multiplication.
 * Design: Saves 50% RAM compared to F32. Uses CMSIS-DSP on hardware.
 */
void matrix_multiply_q15(const MatrixQ15* matrix_a, const MatrixQ15* matrix_b,
                         MatrixQ15* result_matrix);

#endif
