/**
 * @file nn.h
 * @brief Neural Network abstractions for resource-constrained boards.
 *
 * DESIGN DECISIONS:
 * 1. Layer Abstraction: Encapsulates weights, biases, and activation for modularity.
 * 2. Workspace Strategy: Uses "Ping-Pong" buffers to run deep models with 
 *    constant RAM overhead (only 2 intermediate buffers needed).
 * 3. Q15 Support: Parallel infrastructure for quantized (fixed-point) inference.
 */

#ifndef DOT_NN_H
#define DOT_NN_H

#include "matrix.h"

typedef enum {
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID
} ActivationType;

typedef struct {
    const Matrix* weights;
    const Matrix* biases;
    ActivationType activation;
} DenseLayer;

typedef struct {
    const MatrixQ15* weights;
    const MatrixQ15* biases;
    ActivationType activation;
} DenseLayerQ15;

typedef struct {
    const DenseLayer* layers;
    uint16_t layer_count;
} SequentialModel;

/**
 * @brief Standard Dense forward pass.
 */
void dense_layer_forward(const DenseLayer* layer, const Matrix* input_matrix,
                         Matrix* output_matrix);

/**
 * @brief Fixed-point Dense forward pass.
 */
void dense_layer_forward_q15(const DenseLayerQ15* layer, const MatrixQ15* input_matrix,
                             MatrixQ15* output_matrix);

/**
 * @brief Ping-Pong Forward Pass.
 * Design: Alternates between workspace_a and workspace_b to save RAM.
 */
void sequential_forward_with_workspace(const SequentialModel* model, const Matrix* input,
                                       Matrix* workspace_a, Matrix* workspace_b, Matrix* output);

void activation_relu(Matrix* matrix);
void activation_relu_q15(MatrixQ15* matrix);

#endif
