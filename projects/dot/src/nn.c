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

static void add_biases(Matrix* matrix, const Matrix* biases) {
    // Assuming biases is a 1xN or Nx1 vector matching the output columns/rows
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        // Simple element-wise addition (broadcasting logic can be added if needed)
        matrix->data[element_index] += biases->data[element_index];
    }
}

void dense_layer_forward(const DenseLayer* layer, const Matrix* input_matrix,
                         Matrix* output_matrix) {
    // 1. Perform Matrix Multiplication: output = input * weights
    // We use the most performant hardware-aware version
    matrix_multiply(input_matrix, layer->weights, output_matrix);

    // 2. Add Biases
    if (layer->biases != NULL) {
        add_biases(output_matrix, layer->biases);
    }

    // 3. Apply Activation Function
    switch (layer->activation) {
        case ACTIVATION_RELU:
            activation_relu(output_matrix);
            break;
        case ACTIVATION_LINEAR:
        default:
            // Do nothing
            break;
    }
}

void sequential_forward(const SequentialModel* model, const Matrix* input, Matrix* buffers) {
    // 1. First layer takes the initial input
    dense_layer_forward(&model->layers[0], input, &buffers[0]);

    // 2. Subsequent layers take the output of the previous layer as input
    for (uint16_t layer_index = 1; layer_index < model->layer_count; layer_index++) {
        dense_layer_forward(&model->layers[layer_index], &buffers[layer_index - 1],
                            &buffers[layer_index]);
    }
}
