#include "benchmark.h"
#include "matrix.h"
#include "sapi.h"

/**
 * LED Mapping:
 * LED2 -> RED (Processing)
 * LED3 -> GREEN (Done)
 */
#define LED_RED LED2
#define LED_GREEN LED3

#define MATRIX_SIZE 50

// --- Data Buffers in different SRAM banks ---
float data_a[MATRIX_SIZE * MATRIX_SIZE] SRAM_BANK_A;
float data_b[MATRIX_SIZE * MATRIX_SIZE] SRAM_BANK_B;
float data_c[MATRIX_SIZE * MATRIX_SIZE] SRAM_BANK_C;

extern void (*matrix_row_callback)(uint16_t row);

/**
 * Pulse the RED LED (LED2) during calculation.
 */
void visible_toggle_callback(uint16_t row) {
    if (row % 2 == 0) {
        gpioToggle(LED_RED);
    }
}

/**
 * Simple Linear Congruential Generator (LCG) for pseudo-random numbers
 * without standard library dependencies.
 */
static uint32_t next_rand = 1;
uint32_t pseudo_rand(void) {
    next_rand = next_rand * 1103515245 + 12345;
    return (uint32_t)(next_rand / 65536) % 32768;
}

/**
 * Helper to log integers over UART
 */
void log_uint32(uint32_t value) {
    char buffer[11];
    char* ptr = &buffer[10];
    *ptr = '\0';
    if (value == 0) {
        *(--ptr) = '0';
    } else {
        while (value > 0) {
            *(--ptr) = (value % 10) + '0';
            value /= 10;
        }
    }
    uartWriteString(UART_USB, ptr);
}

int main(void) {
    // 1. Initialize Board, UART and Benchmark
    boardConfig();
    uartConfig(UART_USB, 115200);
    benchmark_init();

    uartWriteString(UART_USB, "\r\n--- dot Matrix Stress Test: 50x50 ---\r\n");

    // 2. Initialize input matrices with Random Values using our LCG
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        data_a[i] = (float)(pseudo_rand() % 100) / 10.0f;
        data_b[i] = (float)(pseudo_rand() % 100) / 10.0f;
    }

    Matrix matrix_a = {MATRIX_SIZE, MATRIX_SIZE, data_a};
    Matrix matrix_b = {MATRIX_SIZE, MATRIX_SIZE, data_b};
    Matrix matrix_c = {MATRIX_SIZE, MATRIX_SIZE, data_c};

    // 3. Perform Multiplication with Benchmarking
    uartWriteString(UART_USB, "Calculating 50x50 Multiplication (125k ops)...\r\n");
    matrix_row_callback = visible_toggle_callback;

    benchmark_start();
    matrix_multiply(&matrix_a, &matrix_b, &matrix_c);
    uint32_t total_cycles = benchmark_stop();

    // 4. Output Performance Results
    uartWriteString(UART_USB, "Calculation Finished!\r\n");
    uartWriteString(UART_USB, "Total Clock Cycles: ");
    log_uint32(total_cycles);
    uartWriteString(UART_USB, "\r\n");

    // Verify a sample result
    uartWriteString(UART_USB, "Result[0][0] (Whole part): ");
    log_uint32((uint32_t)data_c[0]);
    uartWriteString(UART_USB, "\r\n");

    gpioWrite(LED_RED, OFF);
    matrix_row_callback = NULL;

    // 5. Success Loop
    while (1) {
        gpioToggle(LED_GREEN);
        delay(500);
    }

    return 0;
}
