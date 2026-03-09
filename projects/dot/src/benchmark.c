#include "benchmark.h"

#ifdef __arm__
#include "sapi.h"

// Core Debug register addresses
#define DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004)
#define DWT_CONTROL (*(volatile uint32_t *)0xE0001000)
#define SCB_DEMCR   (*(volatile uint32_t *)0xE000EDFC)

void benchmark_init(void) {
    // Enable Trace and Debug (DEMCR bit 24)
    SCB_DEMCR |= (1 << 24);
    // Reset cycle counter
    DWT_CYCCNT = 0;
    // Enable cycle counter (DWT_CONTROL bit 0)
    DWT_CONTROL |= (1 << 0);
}

void benchmark_start(void) {
    DWT_CYCCNT = 0;
}

uint32_t benchmark_stop(void) {
    return DWT_CYCCNT;
}

void benchmark_reset(void) {
    DWT_CYCCNT = 0;
}

#else
// Dummy implementation for host-based tests
void benchmark_init(void) {}
void benchmark_start(void) {}
uint32_t benchmark_stop(void) { return 0; }
void benchmark_reset(void) {}
#endif
