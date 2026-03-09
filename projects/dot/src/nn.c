#include "nn.h"

#include <math.h>

void activation_relu(Matrix* matrix) {
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        if (matrix->data[element_index] < 0.0f) {
            matrix->data[element_index] = 0.0f;
        }
    }
}

void activation_relu_q15(MatrixQ15* matrix) {
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        if (matrix->data[element_index] < 0) {
            matrix->data[element_index] = 0;
        }
    }
}

static void add_biases(Matrix* matrix, const Matrix* biases) {
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        matrix->data[element_index] += biases->data[element_index];
    }
}

static void add_biases_q15(MatrixQ15* matrix, const MatrixQ15* biases) {
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        matrix->data[element_index] += biases->data[element_index];
    }
}

void dense_layer_forward(const DenseLayer* layer, const Matrix* input_matrix,
                         Matrix* output_matrix) {
    matrix_multiply(input_matrix, layer->weights, output_matrix);
    if (layer->biases != NULL) {
        add_biases(output_matrix, layer->biases);
    }
    if (layer->activation == ACTIVATION_RELU) {
        activation_relu(output_matrix);
    }
}

void dense_layer_forward_q15(const DenseLayerQ15* layer, const MatrixQ15* input_matrix,
                             MatrixQ15* output_matrix) {
    matrix_multiply_q15(input_matrix, layer->weights, output_matrix);
    if (layer->biases != NULL) {
        add_biases_q15(output_matrix, layer->biases);
    }
    if (layer->activation == ACTIVATION_RELU) {
        activation_relu_q15(output_matrix);
    }
}

/**
 * STRATEGY: One-buffer-per-layer.
 * Fast and simple, but RAM usage scales linearly with the number of layers.
 */
void sequential_forward(const SequentialModel* model, const Matrix* input, Matrix* buffers) {
    dense_layer_forward(&model->layers[0], input, &buffers[0]);
    for (uint16_t layer_index = 1; layer_index < model->layer_count; layer_index++) {
        dense_layer_forward(&model->layers[layer_index], &buffers[layer_index - 1],
                            &buffers[layer_index]);
    }
}

/**
 * STRATEGY: Workspace Ping-Pong.
 * Design: By alternating between two buffers, RAM usage is constant regardless
 * of depth. This enables running deep networks on boards with very little RAM.
 */
void sequential_forward_with_workspace(const SequentialModel* model, const Matrix* input,
                                       Matrix* workspace_a, Matrix* workspace_b, Matrix* output) {
    Matrix* current_input = (Matrix*)input;
    Matrix* current_output = workspace_a;

    for (uint16_t i = 0; i < model->layer_count; i++) {
        if (i == model->layer_count - 1) {
            current_output = output;
        }
        dense_layer_forward(&model->layers[i], current_input, current_output);
        current_input = current_output;
        current_output = (current_output == workspace_a) ? workspace_b : workspace_a;
    }
}
