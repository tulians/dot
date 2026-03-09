#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdint.h>

/**
 * The LPC4337 (Cortex-M4) has a Debug Watchpoint and Trace (DWT) unit
 * that contains a 32-bit cycle counter. This is the most accurate
 * way to measure performance on the board.
 */

// Core Debug registers
#define CORE_DEBUG_DEMCR (*((volatile uint32_t*)0xE000EDFC))
#define CORE_DEBUG_DEMCR_TRCENA (1UL << 24)

// DWT registers
#define DWT_CONTROL (*((volatile uint32_t*)0xE0001000))
#define DWT_CONTROL_CYCCNTENA (1UL << 0)
#define DWT_CYCLE_COUNT (*((volatile uint32_t*)0xE0001004))

/**
 * @brief Enables the hardware cycle counter.
 */
static inline void benchmark_cycle_counter_enable(void) {
    // Enable Trace and Debug block
    CORE_DEBUG_DEMCR |= CORE_DEBUG_DEMCR_TRCENA;
    // Reset cycle counter
    DWT_CYCLE_COUNT = 0;
    // Enable cycle counter
    DWT_CONTROL |= DWT_CONTROL_CYCCNTENA;
}

/**
 * @brief Resets the cycle counter to zero.
 */
static inline void benchmark_cycle_counter_reset(void) {
    DWT_CYCLE_COUNT = 0;
}

/**
 * @brief Returns the current value of the hardware cycle counter.
 */
static inline uint32_t benchmark_get_cycle_count(void) {
    return DWT_CYCLE_COUNT;
}

#endif  // BENCHMARK_H
