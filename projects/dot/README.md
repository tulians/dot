# dot ML Library

**dot** is a high-performance, hardware-aware machine learning library written in C, specifically optimized for the **EDU-CIAA-NXP** (LPC4337, ARM Cortex-M4F). 

The library focuses on providing ultra-fast matrix operations and neural network inference while maintaining a minimal memory footprint and zero dynamic memory allocation.

## 🚀 Key Features

- **Hardware-Aware Memory Banking:** Leverages the LPC4337 AHB bus matrix by striping data across SRAM banks (RAM1, RAM2, RAM3) to prevent bus contention.
- **IKJ Loop Optimization:** Reorders matrix multiplication loops to ensure sequential memory access, significantly reducing bus stalls.
- **4x Loop Unrolling:** Minimizes branch overhead by 75% and maximizes the Cortex-M4F FPU pipeline utilization.
- **Zero-Allocation Inference:** Uses pre-allocated static buffers for intermediate layer results, ensuring deterministic execution and memory safety.
- **On-Target Validation:** Includes an integrated unit test suite that signals success (Blue LED) or failure (Red LED) directly on the hardware.

## 🛠 Technical Decisions

### 1. Memory Architecture (SRAM Striping)
The LPC4337 features multiple SRAM banks connected to different buses. **dot** uses `__attribute__((section(...)))` to place:
- **Matrix A** in SRAM Bank A (RAM2)
- **Matrix B** in SRAM Bank B (RAM3)
- **Results** in SRAM Bank C (RAM1)
This allows the CPU to fetch two operands and write one result in parallel without waiting for the bus.

### 2. Matrix Multiplication Strategies
The library implements several strategies to allow developers to choose the best fit for their needs:
- `ijk_naive`: Standard textbook implementation.
- `ikj`: Reordered for spatial locality.
- `ikj_unrolled_4x`: The performance peak, using pointer arithmetic and 4x unrolling to saturate the FPU.

### 3. Progressive Feedback
Training or inference on embedded devices can be "opaque." **dot** includes a row-level callback system. In the provided example, the **Red LED (LED1)** toggles for every row processed, providing immediate visual confirmation of the CPU's activity.

## 📂 Project Structure

- `src/matrix.c/h`: Core mathematical engine and hardware macros.
- `src/nn.c/h`: Neural network abstractions (Dense Layers, Sequential Models).
- `src/tests.c/h`: On-target unit test suite.
- `src/main.c`: Example application (2-layer MLP).
- `src/benchmark.h`: Cycle-accurate performance measurement utility using the DWT unit.

## 🔨 How to Build

1. Ensure you have the `arm-none-eabi-gcc` toolchain and `openocd` installed.
2. Navigate to the firmware root:
   ```bash
   cd ciaa-ml
   ```
3. Compile the project:
   ```bash
   make
   ```
4. Flash the board:
   ```bash
   make download
   ```

## 🚥 LED Status Codes
- **Blinking Red:** Matrix operation in progress.
- **Solid Blue:** All unit tests passed successfully.
- **Solid Red:** Unit test failure (check your implementation).
- **Solid Green:** Neural network inference completed successfully.
