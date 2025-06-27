#include "CNN.h"
#include <cassert>

// Constructor
CNN::CNN(const hyperparameters& hyper) : _hyp(hyper), FFNN(hyper, 1), L(hyper.filters.size()) {

	// Sets the first conv layer with no channels = 1 (black and white images)
	Conv2DBlock currentConvLayer(1, _hyp.filters[0], hyper);
	m_convLayers.push_back(currentConvLayer);

	// Fill the conv layers with correct dimensions
	for (int l = 0; l < L - 1; l++)
		m_convLayers.push_back(Conv2DBlock(_hyp.filters[l], _hyp.filters[l + 1], hyper));

	// Offsets used for indexing grad vectors and kernel vectors
	m_offset = std::vector<int>(L + 1, 0);
	m_kernelOffset = std::vector<int>(L + 1, 0);
	for (int l = 0; l < L; l++) {
		m_offset[l + 1] = m_offset[l] + _hyp.filters[l];
		m_kernelOffset[l + 1] = m_kernelOffset[l] + (l == 0 ? 1 : _hyp.filters[l - 1]) * _hyp.filters[l];
	}

	// For Adam optimizer
	t = 0;
	for (int l = 0; l < L; l++) {
		std::vector<std::vector<Matrix>> MLayer;
		for (int i = 0; i < m_convLayers[l].getCout(); i++) {
			std::vector<Matrix> MLayerFilters;
			for (int j = 0; j < m_convLayers[l].getCin(); j++)
				MLayerFilters.push_back(Matrix(_hyp.kernel_size, _hyp.kernel_size));
			MLayer.push_back(MLayerFilters);
		}
		M.push_back(MLayer);
		V.push_back(MLayer);
	}

};

// Forward function
Matrix CNN::forward(std::vector<Matrix>& inputs) {
	// A is the number of images
	int A = inputs.size();
	
	// Y & Z are resp. the feature maps before & after activation.
	m_Z = std::vector<Matrix>(A * m_offset[L], Matrix());
	m_Y = m_Z;

//#pragma omp parallel for
	// For each image in the mini-batch, we do the whole forward process.
	m_flattenOutputFtMaps.clear();
	for (int alpha = 0; alpha < A; alpha++) {
		for (int l = 0; l < L; l++) {

			// Forward each output of a layer as input of the next layer
			m_convLayers[l].forward(l == 0 ? std::vector<Matrix>{ inputs[alpha] } : m_convLayers[l - 1].getFeatureMapsReLU()); // forward marche pas.

			// Get certain that sizes fit
			auto featureMaps = m_convLayers[l].getFeatureMaps();
			auto featureMapsActivated = m_convLayers[l].getFeatureMapsReLU();
			assert(featureMaps.size() == _hyp.filters[l] && featureMapsActivated.size() == _hyp.filters[l]);

			// Fill Z with correct values
			for (int j = 0; j < _hyp.filters[l]; j++) {
				std::swap(m_Y[idx(alpha, l, j)], featureMaps[j]);
				std::swap(m_Z[idx(alpha, l, j)], featureMapsActivated[j]);
			}
		}
		// Get the flatten output for each image, to input in the MLP !
		m_flattenOutputFtMaps.push_back(flatten(m_convLayers[L - 1].getFeatureMapsReLU()));
	}

	// Init again the MLP with the right dimensions: change the input dim of the MLP to rows*cols * nbOfFeatureMaps (last convolutional layer dims)
	// I initiate it here because when calling Constructor Method, I can't know the size of the flatten output feature maps. It depends on the size of the initial image.
	if (t == 0) {
		if (_hyp.learn) {
			FFNN = MLP(_hyp, m_flattenOutputFtMaps[0].size());
		}
	}

	return FFNN.forward(m_flattenOutputFtMaps);

};

// Backpropagation & optimization
void CNN::backwards(std::vector<Matrix>& inputs, Matrix y_hot_one) {
	// Backpropagation
	backpropagation(inputs, y_hot_one);

	if (t == 0) print("(mini-batch : loss)");

	// Optimization
	FFNN.Adam();
	Adam();
}

// Backpropagation function (getting dJ/dK by computing dJ/dZ)
void CNN::backpropagation(std::vector<Matrix>& inputs, Matrix& y_hot_one) {
	// A is the number of images
	int A = inputs.size();

	// dY & dZ are resp. the derivates of the feature maps before & after activation.
	m_dZ = std::vector<Matrix>(A * m_offset[L], Matrix());
	m_dY = m_Z;

	Matrix MLP_dX = FFNN.backpropagation(m_flattenOutputFtMaps, y_hot_one);

	// For each image
	for (int alpha = 0; alpha < A; alpha++) {
		// We get the dJ/dZ out of the MLP
		for (int i = 0; i < m_convLayers[L - 1].getCout(); i++)
			m_dZ[idx(alpha, L - 1, i)] = unFlatten(MLP_dX[alpha], i, m_convLayers[L - 1].getFeatureRows(), m_convLayers[L - 1].getFeatureCols());
		
		// For each layer
		for (int l = L - 1; l >= 0; l--) {
			// We compute dJ/dY from dJ/dZ
			assert(_hyp.filters[l] == m_convLayers[l].getCout());
			for (int i = 0; i < m_convLayers[l].getCout(); i++) {
				assert(m_dZ[idx(alpha, l, i)].size() == m_Y[idx(alpha, l, i)].size());
				m_dY[idx(alpha, l, i)] = m_dZ[idx(alpha, l, i)].hadamard(m_Y[idx(alpha, l, i)].derivReLU());
			}

			// If we're not below the first layer, we compute dJ/dZ
			if (l > 0) {
				for (int j = 0; j < m_convLayers[l].getCin(); j++) {
					m_dZ[idx(alpha, l - 1, j)] = Matrix(m_Z[idx(alpha, l - 1, j)].getRows(), m_Z[idx(alpha, l - 1, j)].getCols());
					for (int i = 0; i < m_convLayers[l].getCout(); i++)
						m_dZ[idx(alpha, l - 1, j)] = m_dZ[idx(alpha, l - 1, j)] + m_dY[idx(alpha, l, i)].dilate(_hyp.stride).convolution(m_convLayers[l].getKernels(i, j).rotate180(), _hyp.kernel_size - 1, 1);
					
				}
			}
		}
	}

	// Empty dJ/dK vector before filling it
	m_dK = std::vector<Matrix>(m_kernelOffset[L], Matrix(_hyp.kernel_size, _hyp.kernel_size));

	// For each layer
	for (int l = 0; l < L; l++)
		for (int i = 0; i < m_convLayers[l].getCout(); i++)
			for (int j = 0; j < m_convLayers[l].getCin(); j++)
				// We compute dJ/dK, by summing over all the images a certain convolution
				for (int alpha = 0; alpha < A; alpha++)
					m_dK[idxKernel(l, i, j)] = m_dK[idxKernel(l, i, j)] + (l == 0 ? inputs[alpha] : m_Z[idx(alpha, l - 1, j)]).convolution(m_dY[idx(alpha, l, i)].dilate(_hyp.stride), 0, 1);// * (1.0 / A);
// Here, we can notice that in fact the dK is sometimes 4x4 and not 3x3. It's naive gradient. We remove the last col/row.

}

// Adam optimizer
void CNN::Adam() {
	double beta_m = 0.9;
	double beta_v = 0.999;

	t += 1;
	for (int l = 0; l < L; l++) {
		for (int i = 0; i < m_convLayers[l].getCout(); i++) {
			for (int j = 0; j < m_convLayers[l].getCin(); j++) {
				M[l][i][j] = M[l][i][j] * beta_m + m_dK[idxKernel(l, i, j)] * (1 - beta_m);
				V[l][i][j] = V[l][i][j] * beta_v + m_dK[idxKernel(l, i, j)].hadamard(m_dK[idxKernel(l, i, j)]) * (1 - beta_v);

				Matrix kernelij = m_convLayers[l].getKernels(i, j);

				for (size_t m = 0; m < kernelij.size(); m++) {
					for (size_t n = 0; n < kernelij[0].size(); n++) {
						double M_hat = M[l][i][j][m][n] / (1 - pow(beta_m, t));
						double V_hat = V[l][i][j][m][n] / (1 - pow(beta_v, t));
						kernelij[m][n] = kernelij[m][n] - (M_hat / (sqrt(V_hat) + 1e-8)) * _hyp.learning_rate;
					}
				}
				m_convLayers[l].setKernels(kernelij, i, j);
			}
		}
	}
}

// Index of vectors such as dY or dZ
int CNN::idx(const int& alpha, const int& l, const int& j) {
	return alpha * m_offset[L] + m_offset[l] + j;
}
// Index of vector dK
int CNN::idxKernel(const int& l, const int& i, const int& j) {
	return m_kernelOffset[l] + i * m_convLayers[l].getCin() + j;
}

// Get the output of the CNN.
Matrix CNN::getOutput() {
	return FFNN.getOutput();
}


// Saving the kernels & weights to a .txt
void CNN::saveWeights() {
	FFNN.saveWeights("model_weights.txt");

	std::ofstream file("model_kernels.txt");
	for (auto& layer : m_convLayers) {
		for (int i = 0; i < layer.getCout(); i++) {
			for (int j = 0; j < layer.getCin(); j++) {
				Matrix Kij = layer.getKernels(i, j);
				for (auto& row : Kij) {
					for (double val : row)
						file << val << " ";
					file << "\n";
				}
				file << "===in_channel===\n";
			}
			file << "===out_channel===\n";
		}
		file << "===layer===\n";
	}
	file.close();
}

// Loading the kernels & weights from a .txt
void CNN::loadWeights() {
	FFNN.loadWeights("model_weights.txt");

	std::string filename = "model_kernels.txt";
	std::ifstream file(filename);
    std::string line;
	int layer_index = 0, i = 0, j = 0;
    dmatrix Kij;
    while (std::getline(file, line)) {
        if (line == "===in_channel===") {
            m_convLayers[layer_index].setKernels(Kij, i, j); Kij.clear();
			j++;
        }
		if (line == "===out_channel===") {
			j = 0;
			i++;
		}
		if (line == "===layer===") {
			i = 0, j = 0;
			layer_index++;
		}
        else {
            std::istringstream iss(line);
			dvector row;
			double val;
            while (iss >> val)
                row.push_back(val);
			if(row.size() > 0)
				Kij.push_back(row);
        }
    }
    file.close();
}