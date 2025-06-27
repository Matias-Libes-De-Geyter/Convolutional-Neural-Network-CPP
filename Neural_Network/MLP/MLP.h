#include "..\Blocks\DenseBlock.h"

class MLP {
private:
	hyperparameters _hyperparameters;

	// DenseBlock layers & number of layers
	std::vector<DenseBlock> m_layers;
	int L;

	// Gradient of J (loss function) according to Y = WX + b
	std::vector<Matrix> m_dY;

	// Gradient of J according to weights
	std::vector<Matrix> m_dW;

	// M & V for adam optimizer
	std::vector<Matrix> M, V;

public:
	MLP(const hyperparameters& hyper, const int& input_size);

	Matrix forward(Matrix input);

	Matrix backpropagation(Matrix input, Matrix y_hot_one);
	void Adam();

	Matrix getOutput();
	
	void saveWeights(const std::string& filename);
	void loadWeights(const std::string& filename);
};