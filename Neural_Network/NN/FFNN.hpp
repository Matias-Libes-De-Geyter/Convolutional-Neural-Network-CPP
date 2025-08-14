#include "..\Utilities/functions.hpp"
#include "..\Blocks/DenseBlock.hpp"

#ifndef FFNN_HPP
#define FFNN_HPP


// ======== NEURAL NETWORK ======== //
class FFNN {
private:
	const hyperparameters& _hyper;

	int L;
	std::vector<DenseBlock> m_layers;

	std::vector<Matrix> m_dW;
	std::vector<Matrix> m_dZ;

	Matrix m_output;

public:
	FFNN(const hyperparameters& hyper, int input_size);

	void forward(Matrix& input, const bool learning = false);
	Matrix backpropagation(Matrix& input, const Matrix& y_real);

	void saveWeights(const std::string& filename);
	void loadWeights(const std::string& filename);

	inline std::vector<std::pair<Matrix*, Matrix*>> getParameters() {

		std::vector<std::pair<Matrix*, Matrix*>> weights;
		weights.reserve(L - 1);

		for (int l = 0; l < L - 1; l++)
			weights.emplace_back(&m_layers[l].weights(), &m_dW[l]);

		return weights;
	};

	inline void copyLayers(const FFNN& model) {
		assert(L == model.L);

		for (int l = 0; l < L; ++l) {
			Matrix copy = model.m_layers[l].weights();
			m_layers[l].weights() = copy;
		}
	};

	inline FFNN& operator=(const FFNN& other) {
		if (this == &other)
			return *this;

		L = other.L;
		m_layers.resize(L);
		m_dW.resize(L);
		m_dZ.resize(L);
		for (int l = 0; l < L; ++l) {
			m_layers[l] = other.m_layers[l];
			m_dW[l] = other.m_dW[l];
			m_dZ[l] = other.m_dZ[l];
		}
		m_output = other.m_output;

		return *this;
	};

	inline const Matrix& getOutput() const { return m_output; };
};

#endif