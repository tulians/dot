/**
 * @file benchmark.h
 * @brief High-resolution cycle counting.
 *
 * DESIGN DECISIONS:
 * 1. DWT-based: Uses the ARM Data Watchpoint and Trace unit for clock-cycle precision.
 * 2. Low Overhead: Direct register access to minimize profiling interference.
 */

#ifndef DOT_BENCHMARK_H
#define DOT_BENCHMARK_H

#include <stdint.h>

/**
 * @brief Enables DWT cycle counter on ARM Cortex-M.
 */
void benchmark_init(void);

/**
 * @brief Resets counter to zero.
 */
void benchmark_start(void);

/**
 * @brief Returns elapsed clock cycles.
 */
uint32_t benchmark_stop(void);

#endif
