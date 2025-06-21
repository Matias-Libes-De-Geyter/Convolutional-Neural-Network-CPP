#include "Conv2DBlock.h"
#include <cassert>

// Constructor function
Conv2DBlock::Conv2DBlock(const int& in_channels, const int& out_channels /*(= nb_filters)*/, const hyperparameters& hyper) : _hyp(hyper), _nChannelsIn(in_channels), _nChannelsOut(out_channels) {

	// Xaviers's initialization
	double limit = std::sqrt(6.f / (_nChannelsIn + _nChannelsOut));

	// Kaiming's initialization // Doesn't work well in my implementation
	// limit = std::sqrt(2.f / _nChannelsIn);

	// Kernel initialization
	int kSize = _hyp.kernel_size;
	for (int j = 0; j < _nChannelsOut; j++) {
		for (int i = 0; i < _nChannelsIn; i++) {

			Matrix kernel(kSize, kSize);
			for (int kRow = 0; kRow < kSize; kRow++)
				for (int kCol = 0; kCol < kSize; kCol++)
					kernel[kRow][kCol] = random(-limit, limit);

			// m_filters's "rows" would be filters and "cols" would be kernels.
			m_filters.push_back(kernel);
		}
	}

};

// Activation function used on the feature maps.
static Matrix activate(Matrix& inputs) {
	for (int i = 0; i < inputs.size(); i++)
		for (int j = 0; j < inputs[0].size(); j++)
			inputs[i][j] = std::max(0.0, inputs[i][j]);
	return inputs;
};

// Forward function
void Conv2DBlock::forward(std::vector<Matrix> input, const std::string& softmax) {

	// Get the output dimension of each feat. map
	_featureRows = (input[0].getRows() + 2.0 * _hyp.padding - _hyp.kernel_size) / _hyp.stride + 1;
	_featureCols = (input[0].getCols() + 2.0 * _hyp.padding - _hyp.kernel_size) / _hyp.stride + 1;

	// Create the empty feature maps
	m_featureMaps = std::vector<Matrix>(_nChannelsOut, Matrix(_featureRows, _featureCols));
	m_featureMapsReLU = m_featureMaps;
	
	// Get sure the input has the right size to go in the conv. layer
	assert(input.size() == _nChannelsIn);

	// Convolution Layer
	for (int i = 0; i < _nChannelsOut; i++)
		for (int j = 0; j < _nChannelsIn; j++)					
			m_featureMaps[i] = m_featureMaps[i] + input[j].convolution(m_filters[i * _nChannelsIn + j], _hyp.padding, _hyp.stride);// * (1.0 / _nChannelsIn);
	/// SEEMS THAT MY LOSS FUNCTION DECREASES WAY FASTER WITHOUT THE NORMALIZATION HERE AND IN CNN.CPP ("1/A")...

	// Activation Layer
	for (int j = 0; j < _nChannelsOut; j++)
		m_featureMapsReLU[j] = activate(m_featureMaps[j]);

};

// Set and Get kernels.
void Conv2DBlock::setKernels(const Matrix& newKernel, const int& i, const int& j) {
	m_filters[i * _nChannelsIn + j] = newKernel;
}
Matrix Conv2DBlock::getKernels(const int& i, const int& j) {
	return m_filters[i * _nChannelsIn + j];
}

std::vector<Matrix> Conv2DBlock::getFeatureMaps() { return m_featureMaps; };
std::vector<Matrix> Conv2DBlock::getFeatureMapsReLU() { return m_featureMapsReLU; };

int Conv2DBlock::getFeatureRows() const { return _featureRows; }
int Conv2DBlock::getFeatureCols() const { return _featureCols; }

const int Conv2DBlock::getCin() const { return _nChannelsIn; }
const int Conv2DBlock::getCout() const { return _nChannelsOut; }
