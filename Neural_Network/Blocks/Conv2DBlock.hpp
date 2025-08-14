#include "..\Utilities/functions.hpp"


#ifndef CB_HPP
#define CB_HPP

class Conv2DBlock {
private:
	std::vector<Matrix> m_filters;
	std::vector<Matrix> m_featureMaps;
	std::vector<Matrix> m_featureMapsReLU;
	const hyperparameters _hyp;
	const int _nChannelsIn;
	const int _nChannelsOut;
	int _featureRows;
	int _featureCols;

public:
	Conv2DBlock(const int& in_channels, const int& out_channels, const hyperparameters& hyp);
	void forward(std::vector<Matrix>& input, const std::string& softmax = "");

	// Set and Get kernels.
	inline void setKernels(const Matrix& newKernel, const int i, const int j) { m_filters[i * _nChannelsIn + j] = newKernel; }
	inline Matrix& getKernels(const int i, const int j) { return m_filters[i * _nChannelsIn + j]; }
	inline const Matrix& getKernels(const int i, const int j) const { return m_filters[i * _nChannelsIn + j]; }

	std::vector<Matrix> getFeatureMaps();
	std::vector<Matrix> getFeatureMapsReLU();
	int getFeatureRows() const;
	int getFeatureCols() const;
	const int getCin() const;
	const int getCout() const;
};

#endif