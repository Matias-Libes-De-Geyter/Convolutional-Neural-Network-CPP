#include "CNN.hpp"
#include <cassert>

// Constructor
CNN::CNN(const hyperparameters& hyper) : _hyp(hyper), _outputFFNN(hyper, 1), L(hyper.filters.size()) {

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

	first_iteration = true;
}

// Forward function
void CNN::forward(const std::vector<Matrix>& inputs, const bool learning) {
	// A is the number of images
	int A = inputs.size();

	// Y & Z are resp. the feature maps before & after activation.
	m_Z = std::vector<Matrix>(A * m_offset[L], Matrix());
	m_Y = m_Z;

	// For each image in the mini-batch, we do the whole forward process.
	d_matrix flattenCNNOutput;
	for (int alpha = 0; alpha < A; alpha++) {
		for (int l = 0; l < L; l++) {

			// Forward each output of a layer as input of the next layer
			std::vector<Matrix> forward_val = (l == 0 ? std::vector<Matrix>{ inputs[alpha] } : m_convLayers[l - 1].getFeatureMapsReLU());
			m_convLayers[l].forward(forward_val);

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
		flattenCNNOutput.push_back(flatten(m_convLayers[L - 1].getFeatureMapsReLU()));
	}

	// Init again the MLP with the right dimensions: change the input dim of the MLP to rows*cols * nbOfFeatureMaps (last convolutional layer dims)
	// I initiate it here because when calling Constructor Method, I can't know the size of the flatten output feature maps. It depends on the size of the initial image.
	if (first_iteration) {
		first_iteration = false;
		if (learning)
			_outputFFNN = FFNN(_hyp, flattenCNNOutput[0].size());
	}

	m_flattenOutputFtMaps = flattenCNNOutput;
	_outputFFNN.forward(m_flattenOutputFtMaps, learning);

};

// Backpropagation function (getting dJ/dK by computing dJ/dZ)
void CNN::backpropagation(const std::vector<Matrix>& inputs, const Matrix& y_real) {
	// A is the number of images
	int A = inputs.size();

	// dY & dZ are resp. the derivates of the feature maps before & after activation.
	m_dZ = std::vector<Matrix>(A * m_offset[L], Matrix());
	m_dY = m_Z;

	Matrix MLP_dX = _outputFFNN.backpropagation(m_flattenOutputFtMaps, y_real);

	// For each image
	for (int alpha = 0; alpha < A; alpha++) {
		// We get the dJ/dZ out of the MLP
		for (int i = 0; i < m_convLayers[L - 1].getCout(); i++)
			m_dZ[idx(alpha, L - 1, i)] = unFlatten(MLP_dX.row(alpha), i, m_convLayers[L - 1].getFeatureRows(), m_convLayers[L - 1].getFeatureCols());
		
		// For each layer
		for (int l = L - 1; l >= 0; l--) {
			// We compute dJ/dY from dJ/dZ
			assert(_hyp.filters[l] == m_convLayers[l].getCout());
			for (int i = 0; i < m_convLayers[l].getCout(); i++) {
				assert(m_dZ[idx(alpha, l, i)].rows() == m_Y[idx(alpha, l, i)].rows());
				m_dY[idx(alpha, l, i)] = m_dZ[idx(alpha, l, i)].hadamard(ACTIVATION::deriv_ReLU(m_Y[idx(alpha, l, i)]));
			}

			// If we're not below the first layer, we compute dJ/dZ
			if (l > 0) {
				for (int j = 0; j < m_convLayers[l].getCin(); j++) {
					m_dZ[idx(alpha, l - 1, j)] = Matrix(m_Z[idx(alpha, l - 1, j)].rows(), m_Z[idx(alpha, l - 1, j)].cols());
					for (int i = 0; i < m_convLayers[l].getCout(); i++)
						m_dZ[idx(alpha, l - 1, j)] += Matrix::convolution(m_dY[idx(alpha, l, i)].dilate(_hyp.stride), m_convLayers[l].getKernels(i, j).rotate180(), _hyp.kernel_size - 1, 1);
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
					m_dK[idxKernel(l, i, j)] += Matrix::convolution((l == 0 ? inputs[alpha] : m_Z[idx(alpha, l - 1, j)]), m_dY[idx(alpha, l, i)].dilate(_hyp.stride), 0, 1);// * (1.0 / A);

// Here, we can notice that in fact the dK is sometimes 4x4 and not 3x3. It's naive gradient. We remove the last col/row.

}

// Saving the kernels & weights to a .txt
void CNN::saveWeights(const std::string& filenameW, const std::string& filenameK) {
	_outputFFNN.saveWeights(filenameW);

	std::ofstream file(filenameK);
	for (auto& layer : m_convLayers) {
		for (int i = 0; i < layer.getCout(); i++) {
			for (int j = 0; j < layer.getCin(); j++) {
				Matrix Kij = layer.getKernels(i, j);
				for (size_t i = 0; i < Kij.rows(); i++) {
					for (size_t j = 0; j < Kij.cols(); j++)
						file << Kij(i, j) << " ";
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
void CNN::loadWeights(const std::string& filenameW, const std::string& filenameK) {
	_outputFFNN.loadWeights(filenameW);

	std::ifstream file(filenameK);
    std::string line;
	int layer_index = 0, i = 0, j = 0;
    d_matrix Kij;
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
			d_vector row;
			double val;
            while (iss >> val)
                row.push_back(val);
			if(row.size() > 0)
				Kij.push_back(row);
        }
    }
    file.close();
}