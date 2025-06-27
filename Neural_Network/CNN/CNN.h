#include "..\Blocks\Conv2DBlock.h"
#include "..\MLP\MLP.h"


class CNN {
private:
	hyperparameters _hyp;
	MLP FFNN;

	// Conv2DBlock layers & number of layers
	std::vector<Conv2DBlock> m_convLayers;
	const int L;

	// Output of the conv layers, input of the MLP
	dmatrix m_flattenOutputFtMaps;

	// Backpropagation variables
	std::vector<Matrix> m_Z;
	std::vector<Matrix> m_Y;
	std::vector<Matrix> m_dZ;
	std::vector<Matrix> m_dY;
	std::vector<Matrix> m_dK;

	// Adam variables
	std::vector<std::vector<std::vector<Matrix>>> M, V, v;
	int t;

	// Vectors indexing
	std::vector<int> m_offset;
	std::vector<int> m_kernelOffset;
	int idx(const int& alpha, const int& l, const int& j);
	int idxKernel(const int& l, const int& i, const int& j);

public:
	CNN(const hyperparameters& hyper);

	Matrix forward(std::vector<Matrix>& inputs);
	
	void backwards(std::vector<Matrix>& inputs, Matrix y_hot_one);
	void backpropagation(std::vector<Matrix>& inputs, Matrix& y_hot_one);
	void Adam();

	Matrix getOutput();
	
	void loadWeights();
	void saveWeights();
};