#include "MLP.h"

MLP::MLP(const hyperparameters& hyper, const int& input_size) : _hyperparameters(hyper), L(_hyperparameters.hidden_layers_sizes.size() + 1) {

    // Init gradients to 0
    m_dY = std::vector<Matrix>(L);
    m_dW = m_dY;

    // Add the input size to the layers, and initiate all layers into "m_layers"
    // Filling M & V vectors for Adam optimizer
    std::vector<int> hidden_layers_sizes = _hyperparameters.hidden_layers_sizes;
    hidden_layers_sizes.insert(hidden_layers_sizes.begin(), input_size);
    hidden_layers_sizes.push_back(_hyperparameters.number_classes);
    for (int l = 0; l < L; l++) {
        DenseBlock layer(hidden_layers_sizes[l], hidden_layers_sizes[l + 1]);
        m_layers.push_back(layer);

        M.push_back(Matrix(hidden_layers_sizes[l] + 1, hidden_layers_sizes[l + 1]));
        V.push_back(M[l]);
    }

};

// Forward function
Matrix MLP::forward(Matrix input) {

    // Forwarding the first layer, then passing it through all the layers. The last layer is activated with a softmax.
    m_layers[0].forward(input);
    for (int l = 1; l < L - 1; l++)
        m_layers[l].forward(_hyperparameters.learn ? m_layers[l - 1].output().dropoutMask(_hyperparameters.dropout_rate) : m_layers[l - 1].output());
    m_layers[L - 1].forward(m_layers[L - 2].output(), "softmax");

    return m_layers.back().output();

}

// Computation of gradient of J (loss function) according to weights.
Matrix MLP::backpropagation(Matrix input, Matrix y_hot_one) {

    // Getting values at the output of the MLP
    m_dY[L - 1] = m_layers.back().output() - y_hot_one;
    m_dW[L - 1] = m_layers[L - 2].output().addBias_then_T() * m_dY[L - 1];

    // Going backwards to get all the gradients from the output
    for (int l = L - 2; l >= 0; l--) {
        m_dY[l] = (m_dY[l + 1] * m_layers[l + 1].getWeights().T_then_removeBias()).hadamard(m_layers[l].preactivation().derivReLU());
        m_dW[l] = (l == 0 ? input : m_layers[l - 1].output()).addBias_then_T() * m_dY[l];
    }

    // Returning the gradient of J according to X, at the input of the MLP (backprop. of the CNN will start with this data)
    return (m_dY[0] * m_layers[0].getWeights().T_then_removeBias()).hadamard(input.derivReLU());

}

// Adam optimizer for the MLP, updating the gradients.
void MLP::Adam() {
    double beta_m = 0.9;
    double beta_v = 0.999;

    static int t = 0;
    t += 1;

    for (int l = 0; l < L; l++) {
        M[l] = M[l] * beta_m + m_dW[l] * (1 - beta_m);
        V[l] = V[l] * beta_v + m_dW[l].hadamard(m_dW[l]) * (1 - beta_v);

        Matrix weight = m_layers[l].getWeights();
        for (size_t i = 0; i < weight.size(); i++) {
            for (size_t j = 0; j < weight[0].size(); j++) {
                double M_hat = M[l][i][j] / (1 - pow(beta_m, t));
                double V_hat = V[l][i][j] / (1 - pow(beta_v, t));
                weight[i][j] = weight[i][j] - (M_hat / (sqrt(V_hat) + 1e-8)) * _hyperparameters.learning_rate;
            }
        }
        m_layers[l].setWeights(weight);
    }

}

// Get the output of the MLP.
Matrix MLP::getOutput() {
    return m_layers.back().output();
}




// Saving the weights to a .txt
void MLP::saveWeights(const std::string& filename) {
    std::ofstream file(filename);
    for (auto& layer : m_layers) {
        Matrix W = layer.getWeights();
        for (auto& row : W) {
            for (double val : row)
                file << val << " ";
            file << "\n";
        }
        file << "===\n"; // Séparateur entre couches
    }
    file.close();
}

// Loading the weights from a .txt
void MLP::loadWeights(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    dmatrix W;
    int layer_index = 0;
    int inputSize;
    while (std::getline(file, line)) {
        if (line == "===") {
            // Basically, recreate the MLP with the right input dimension.
            if (layer_index == 0) {
                inputSize = W.size();
                *this = MLP(_hyperparameters, inputSize - 1);
            }
            m_layers[layer_index].setWeights(W);
            W.clear();
            layer_index++;
        }
        else {
            std::istringstream iss(line);
            dvector row;
            double val;
            while (iss >> val)
                row.push_back(val);
            W.push_back(row);
        }
    }
    file.close();

}
