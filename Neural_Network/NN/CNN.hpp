#include "..\Blocks/Conv2DBlock.hpp"
#include "FFNN.hpp"


#ifndef CNN_HPP
#define CNN_HPP

class CNN {
private:
	hyperparameters _hyp;
	FFNN _outputFFNN;

	// Conv2DBlock layers & number of layers
	std::vector<Conv2DBlock> m_convLayers;
	const int L;

	// Output of the conv layers, input of the MLP
	Matrix m_flattenOutputFtMaps;

	// Backpropagation variables
	std::vector<Matrix> m_Z;
	std::vector<Matrix> m_Y;
	std::vector<Matrix> m_dZ;
	std::vector<Matrix> m_dY;
	std::vector<Matrix> m_dK;

    bool first_iteration;

	// Vectors indexing
	std::vector<int> m_offset;
	std::vector<int> m_kernelOffset;

    // Index of vectors such as dY or dZ
    inline int idx(const int& alpha, const int& l, const int& j) { return alpha * m_offset[L] + m_offset[l] + j; };
    // Index of vector dK
    inline int idxKernel(const int& l, const int& i, const int& j) { return m_kernelOffset[l] + i * m_convLayers[l].getCin() + j; };

public:
	CNN(const hyperparameters& hyper);

	void forward(const std::vector<Matrix>& inputs, const bool learning);

	void backpropagation(const std::vector<Matrix>& inputs, const Matrix& y_real);

    void saveWeights(const std::string& filenameW, const std::string& filenameK);
    void loadWeights(const std::string& filenameW, const std::string& filenameK);

    inline std::vector<std::pair<Matrix*, Matrix*>> getParameters() {
        std::vector<std::pair<Matrix*, Matrix*>> params;

        int gradIndex = 0;
        for (size_t l = 0; l < m_convLayers.size(); ++l) {
            const int Cin = m_convLayers[l].getCin();
            const int Cout = m_convLayers[l].getCout();

            for (int i = 0; i < Cout; ++i)
                for (int j = 0; j < Cin; ++j)
                    params.emplace_back(&m_convLayers[l].getKernels(i, j), &m_dK[gradIndex++]);
        }
        auto ffnn_params = _outputFFNN.getParameters();
        params.insert(params.end(), ffnn_params.begin(), ffnn_params.end());

        return params;
    };

    inline void copyLayers(const CNN& model) {

        for (size_t l = 0; l < m_convLayers.size(); l++) {
            const int Cin = m_convLayers[l].getCin();
            const int Cout = m_convLayers[l].getCout();

            for (int i = 0; i < Cout; ++i) {
                for (int j = 0; j < Cin; ++j) {
                    Matrix copy = model.m_convLayers[l].getKernels(i, j);
                    m_convLayers[l].setKernels(copy, i, j);
                }
            }
        }

        _outputFFNN.copyLayers(model._outputFFNN);
    };

    inline const Matrix& getOutput() const { return _outputFFNN.getOutput(); };

};

#endif