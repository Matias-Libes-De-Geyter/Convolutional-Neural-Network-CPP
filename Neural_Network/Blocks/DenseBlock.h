#include "..\Utilities\functions.h"


class DenseBlock {
private:
	Matrix m_weights;
	Matrix m_preactivation;
	Matrix m_output;

	dvector delta;
	
public:
	DenseBlock(const int& n_inputs, const int& n_neurons);
	void forward(Matrix inputs, const std::string& softmax = "");

	void setWeights(const Matrix& weights);
	Matrix getWeights();
	
	Matrix preactivation();
	Matrix output();
};