#ifndef BENCHMARK_H
#define DOT_BENCHMARK_H

#include <stdint.h>

/**
 * @brief Initializes the DWT (Data Watchpoint and Trace) unit for cycle counting.
 * Only functional on ARM Cortex-M architecture.
 */
void benchmark_init(void);

/**
 * @brief Resets and starts the cycle counter.
 */
void benchmark_start(void);

/**
 * @brief Stops the cycle counter and returns the elapsed cycles.
 * @return uint32_t Number of clock cycles since benchmark_start() was called.
 */
uint32_t benchmark_stop(void);

/**
 * @brief Resets the cycle counter to zero.
 */
void benchmark_reset(void);

#endif  // DOT_BENCHMARK_H
