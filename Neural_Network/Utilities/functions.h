#include "Matrix.h"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define EULERS_NUMBER pow((1.0 + 1.0 / 10000000.0), 10000000.0)

// Hyperparameters
struct hyperparameters {
	int number_classes;
	std::vector<int> hidden_layers_sizes;
	int epochs;
	int mini_batch_size;
	double learning_rate;
	double dropout_rate;
	bool early_stopping;
	int patience;
	bool learn;
	bool test;
	bool store_weights;
	std::vector<int> filters;
	int kernel_size;
	int padding;
	int stride;
	int img_size;
	std::string database;
};

double random(const double& min, const double& max); // Random function
Matrix hotOne(const dvector& y, const int& nElements); // Returns the "hot one" matrix of a vector.
double CELossFunction(const Matrix& y_pred, const Matrix& y_true); // Return the cross-entropy loss.

// Flatten and unflatten functions
dvector flatten(const std::vector<Matrix>& A);
Matrix unFlatten(const dvector& A, const int& iFtMap, const int& rows, const int& cols);

// Utility function used in TrainerClassifier.h
Matrix flattenToMatrix(const std::vector<double>& flat_image, int rows, int cols);
void readMNIST(const std::string& imageFile, const std::string& labelFile, dmatrix& images, dvector& labels);


// ===== PRINT FUNCTIONS ===== //

// Print other stuff
template<typename... Args> void print(const Args&... args) { (std::cout << ... << args) << std::endl; }

// Print matrices & vectors
template<typename T>
void print(const T& container) {
	if constexpr (std::is_same_v<T, Matrix> || std::is_same_v<T, dvector>) {
		std::cout << "[";
		bool first = true;
		for (auto element : container) {
			if constexpr (!std::is_same_v<T, Matrix>) std::cout << (!first ? ", " : "") << element;
			else print(element);
			first = false;
		}
		std::cout << "]," << std::endl;
	}
	else print(container, "");
}

#endif