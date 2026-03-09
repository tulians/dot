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
    // Assuming biases is a 1xN or Nx1 vector matching the output columns/rows
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        // Simple element-wise addition (broadcasting logic can be added if needed)
        matrix->data[element_index] += biases->data[element_index];
    }
}

static void add_biases_q15(MatrixQ15* matrix, const MatrixQ15* biases) {
    uint32_t total_elements = (uint32_t)matrix->rows * matrix->columns;
    for (uint32_t element_index = 0; element_index < total_elements; element_index++) {
        // Fixed point addition
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

void dense_layer_forward_q15(const DenseLayerQ15* layer, const MatrixQ15* input_matrix,
                             MatrixQ15* output_matrix) {
    // 1. Perform Q15 Matrix Multiplication
    matrix_multiply_q15(input_matrix, layer->weights, output_matrix);

    // 2. Add Biases
    if (layer->biases != NULL) {
        add_biases_q15(output_matrix, layer->biases);
    }

    // 3. Apply Activation Function
    switch (layer->activation) {
        case ACTIVATION_RELU:
            activation_relu_q15(output_matrix);
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

void sequential_forward_with_workspace(const SequentialModel* model, const Matrix* input,
                                       Matrix* workspace_a, Matrix* workspace_b, Matrix* output) {
    Matrix* current_input = (Matrix*)input;
    Matrix* current_output = workspace_a;

    for (uint16_t i = 0; i < model->layer_count; i++) {
        // If it's the last layer, write directly to final output
        if (i == model->layer_count - 1) {
            current_output = output;
        }

        dense_layer_forward(&model->layers[i], current_input, current_output);

        // Ping-pong pointers for next layer
        current_input = current_output;
        current_output = (current_output == workspace_a) ? workspace_b : workspace_a;
    }
}
