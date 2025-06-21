#include "functions.h"

// Random function
double random(const double& min, const double& max) {
	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}
// Function that gets the hot one matrix out of a vector
Matrix hotOne(const dvector& y, const int& nElements) {

	Matrix C(y.size(), nElements);

	for (int i = 0; i < y.size(); i++)
		C[i][y[i]] = 1;

	return C;
}
// Cross-entropy loss computation
double CELossFunction(const Matrix& y_pred, const Matrix& y_true) {
	double mean_loss = 0.0;

	for (int i = 0; i < y_pred.size(); i++)
		for (int j = 0; j < y_pred[0].size(); j++)
			if (y_true[i][j] == 1.0) {
				mean_loss += -log(std::max(y_pred[i][j], 1e-9));
				break;
			}

	mean_loss /= y_pred.size();

	return mean_loss;
}


// Flatten and unflatten functions
dvector flatten(const std::vector<Matrix>& A) {
	dvector C;
	for (int i = 0; i < A.size(); i++)
		for (int r = 0; r < A[i].getRows(); r++)
			for (int c = 0; c < A[i].getCols(); c++)
				C.push_back(A[i][r][c]);

	return C;
}
Matrix unFlatten(const dvector& A, const int& iFtMap, const int& rows, const int& cols) {

	Matrix currentMatrix(rows, cols);
	for (int r = 0; r < rows; r++)
		for (int c = 0; c < cols; c++)
			currentMatrix[r][c] = A[iFtMap * (rows * cols) + r * cols + c];

	return currentMatrix;
}



// Flatten function
Matrix flattenToMatrix(const std::vector<double>& flat_image, int rows, int cols) {
	Matrix mat(rows, cols);

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			mat[i][j] = flat_image[i * cols + j];
	return mat;
}
// Load the MNIST database
int reverseInt(int i) { // To little-endian function
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void readMNIST(const std::string& imageFile, const std::string& labelFile,
	dmatrix& images, dvector& labels) {
	std::ifstream imgFile(imageFile, std::ios::binary);
	std::ifstream lblFile(labelFile, std::ios::binary);

	int magicNumber, numImages, numRows, numCols;
	imgFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	imgFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
	numImages = reverseInt(numImages);
	imgFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
	numRows = reverseInt(numRows);
	imgFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
	numCols = reverseInt(numCols);

	int labelMagicNumber, numLabels;
	lblFile.read(reinterpret_cast<char*>(&labelMagicNumber), sizeof(labelMagicNumber));
	labelMagicNumber = reverseInt(labelMagicNumber);
	lblFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
	numLabels = reverseInt(numLabels);

	int n = std::min(numImages, numLabels);
	images.resize(n, std::vector<double>(numRows * numCols));
	labels.resize(n);

	for (int i = 0; i < n; ++i) {
		// Images
		for (int j = 0; j < numRows * numCols; ++j) {
			unsigned char pixel;
			imgFile.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			images[i][j] = static_cast<double>(pixel) / 255.0;
		}
		// Labels
		unsigned char label;
		lblFile.read(reinterpret_cast<char*>(&label), sizeof(label));
		labels[i] = static_cast<double>(label);
	}
}