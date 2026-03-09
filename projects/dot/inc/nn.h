/**
 * @file dot_nn.h
 * @brief part of the dot ML library.
 * Neural Network abstractions for the EDU-CIAA-NXP.
 */

#ifndef DOT_NN_H
#define DOT_NN_H

#include "matrix.h"

/**
 * @brief Types of available activation functions.
 */
typedef enum {
    ACTIVATION_LINEAR, /**< No change (Output = Input) */
    ACTIVATION_RELU,   /**< Rectified Linear Unit (f(x) = max(0, x)) */
    ACTIVATION_SIGMOID /**< Sigmoid function (1 / (1 + e^-x)) */
} ActivationType;

/**
 * @brief Represents a single Dense (Fully Connected) Layer.
 *
 * This structure links weight and bias matrices with a specific activation
 * function to define the behavior of a single layer of a neural network.
 */
typedef struct {
    const Matrix* weights;     /**< Weight matrix (InputSize x OutputSize) */
    const Matrix* biases;      /**< Bias matrix (1 x OutputSize) */
    ActivationType activation; /**< Activation function to apply after multiplication */
} DenseLayer;

/**
 * @brief Represents a single Dense (Fully Connected) Layer in Q15.
 */
typedef struct {
    const MatrixQ15* weights;  /**< Weight matrix (InputSize x OutputSize) */
    const MatrixQ15* biases;   /**< Bias matrix (1 x OutputSize) */
    ActivationType activation; /**< Activation function to apply after multiplication */
} DenseLayerQ15;

/**
 * @brief Represents a Sequential model composed of a stack of layers.
 *
 * A sequential model provides a clear structure for running an input
 * through several neural network layers in a predefined order.
 */
typedef struct {
    const DenseLayer* layers; /**< Pointer to an array of layers */
    uint16_t layer_count;     /**< Total number of layers in the model */
} SequentialModel;

/**
 * @brief Performs a forward pass through a single Dense Layer.
 *
 * Calculates the output using the formula: output = activation(input * weights + bias).
 *
 * @param layer The dense layer configuration (weights, biases, activation).
 * @param input_matrix The input vector or matrix (1 x InputSize).
 * @param output_matrix Pre-allocated matrix for the output (1 x OutputSize).
 */
void dense_layer_forward(const DenseLayer* layer, const Matrix* input_matrix,
                         Matrix* output_matrix);

/**
 * @brief Performs a forward pass through a single Q15 Dense Layer.
 */
void dense_layer_forward_q15(const DenseLayerQ15* layer, const MatrixQ15* input_matrix,
                             MatrixQ15* output_matrix);

/**
 * @brief Performs a forward pass through a complete Sequential model.
 *
 * Iteratively calculates the output for each layer, feeding the result
 * of one layer as the input for the next. This function requires a
 * pre-allocated array of result buffers to avoid dynamic memory usage.
 *
 * @param model The sequential model configuration.
 * @param input The initial input to the network.
 * @param buffers Array of Matrix structures for intermediate/final results.
 *                Must contain exactly 'layer_count' pre-allocated matrices.
 */
void sequential_forward(const SequentialModel* model, const Matrix* input, Matrix* buffers);

/**
 * @brief Performs a forward pass using only TWO intermediate buffers.
 *
 * This "Ping-Pong" strategy reuses memory by alternating between two workspace
 * buffers, drastically reducing RAM requirements for deep models.
 *
 * @param model The sequential model configuration.
 * @param input Initial input matrix.
 * @param workspace_a First temporary buffer (must fit largest layer output).
 * @param workspace_b Second temporary buffer (must fit largest layer output).
 * @param output Final output matrix.
 */
void sequential_forward_with_workspace(const SequentialModel* model, const Matrix* input,
                                       Matrix* workspace_a, Matrix* workspace_b, Matrix* output);

/**
 * @brief In-place ReLU activation.
 *
 * Clips all values less than 0.0 to exactly 0.0.
 *
 * @param matrix Matrix to apply the activation to.
 */
void activation_relu(Matrix* matrix);

/**
 * @brief In-place Q15 ReLU activation.
 */
void activation_relu_q15(MatrixQ15* matrix);

#endif  // NN_H
