#include "Matrix.hpp"

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init) {
	_rows = init.size();
	_cols = init.begin()->size();

	_matrix.clear();
	_matrix.reserve(_rows * _cols);

	for (const auto& row : init)
		_matrix.insert(_matrix.end(), row.begin(), row.end());
}
Matrix::Matrix(std::vector<std::vector<double>> init) {
	_rows = init.size();
	_cols = init.begin()->size();

	_matrix.clear();
	_matrix.reserve(_rows * _cols);

	for (const auto& row : init)
		_matrix.insert(_matrix.end(), row.begin(), row.end());
}
Matrix& Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {
	_rows = init.size();
	_cols = init.begin()->size();

	_matrix.clear();
	_matrix.reserve(_rows * _cols);

	for (auto& row : init)
		_matrix.insert(_matrix.end(), row.begin(), row.end());

	return *this;
}

Matrix Matrix::operator*(const Matrix& B) const {
	assert(_cols == B.rows());

	size_t new_cols = B.cols();
	Matrix C(_rows, new_cols);

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t k = 0; k < _cols; k++) {
			double Aik = _matrix[row_offset + k];
			for (size_t j = 0; j < new_cols; j++)
				C(i, j) += Aik * B(k, j);
		}
	}

	return C;
}

Matrix Matrix::operator*(const double b) const {

	Matrix C(_rows, _cols);

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] * b;
	}

	return C;
}
Matrix& Matrix::operator*=(const double b) {

	for (size_t idx = 0; idx < _rows * _cols; idx++)
		_matrix[idx] *= b;

	return *this;
}

Matrix Matrix::hadamard(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] * B(i, j);
	}

	return C;
}
const bool Matrix::operator==(const Matrix& B) const {
	if (_rows == B.rows()) {
		if (_cols == B.cols()) {
			for (size_t idx = 0; idx < _rows * _cols; idx++)
				if (B(idx) != _matrix[idx])
					return false;
			return true;
		}
	}

	return false;
}


Matrix Matrix::operator+(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] + B(i, j);
	}

	return C;
}

Matrix& Matrix::operator+=(const Matrix& B) {
	// I removed this because of the backprop m_dK calculation in Backpropagation: "Here, we can notice that in fact the dK is sometimes 4x4 and not 3x3. It's naive gradient. We remove the last col/row."
	//assert(_rows == B.rows());
	//assert(_cols == B.cols());

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			_matrix[row_offset + j] += B(i, j);
	}

	return *this;
}

Matrix Matrix::operator-(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] - B(i, j);
	}

	return C;
}

Matrix& Matrix::operator-=(const Matrix& B) {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			_matrix[row_offset + j] -= B(i, j);
	}

	return *this;
}



Matrix Matrix::T() const {

	Matrix C(_cols, _rows);
	for (size_t j = 0; j < _rows; j++) {
		size_t row_offset = j * _cols;
		for (size_t i = 0; i < _cols; i++)
			C(i, j) = _matrix[row_offset + i];
	}

	return C;
}
Matrix Matrix::addBias() const {

	Matrix C(_rows, _cols + 1);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j];
		C(i, _cols) = 1;
	}

	return C;
}
Matrix Matrix::addBias_then_T() const {

	Matrix C(_cols + 1, _rows);
	for (size_t j = 0; j < _rows; j++) {
		size_t row_offset = j * _cols;
		for (size_t i = 0; i < _cols; i++)
			C(i, j) = _matrix[row_offset + i];
		C(_cols, j) = 1;
	}

	return C;
}
Matrix Matrix::removeBias() const {

	Matrix C(_rows - 1, _cols);
	for (size_t i = 0; i < _rows - 1; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j];
	}

	return C;
}
Matrix Matrix::T_then_removeBias() const {

	Matrix C(_cols, _rows - 1);
	for (size_t i = 0; i < _cols; i++)
		for (size_t j = 0; j < _rows - 1; j++)
			C(i, j) = (*this)(j, i);

	return C;
}

Matrix Matrix::dropoutMask(double dropout) const {
	double keep_prob = 1.0 - dropout;

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; ++i) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; ++j)
			C(i, j) = ((double)rand() / RAND_MAX > keep_prob) ? 0.0 : (_matrix[row_offset + j] / keep_prob);
	}

	return C;
}

Matrix Matrix::setMaxToOne() const {


	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;

		double max_value = _matrix[row_offset];
		size_t max_index = 0;
		for (size_t j = 0; j < _cols; j++) {
			double element = _matrix[row_offset + j];
			if (element > max_value) {
				max_value = element;
				max_index = j;
			}
		}
		C(i, max_index) = 1;
	}

	return C;
}

int Matrix::getMaxIndex() const {

	assert(_rows == 1);

	double max = _matrix[0];
	int index = 0;
	for (size_t j = 1; j < _cols; j++) {
		double val = _matrix[j];
		if (val > max) {
			max = val;
			index = j;
		}
	}
	return index;
}

void Matrix::fill(const double b) {

	for (size_t idx = 0; idx < _rows * _cols; idx++)
		_matrix[idx] = b;
}

Matrix Matrix::convolution(const Matrix& input, const Matrix& kernel, const int padding, const int stride) {
	assert(kernel.rows() < input.rows() && kernel.cols() < input.cols());

	size_t out_rows = (input.rows() + 2 * padding - kernel.rows()) / stride + 1;
	size_t out_cols = (input.cols() + 2 * padding - kernel.cols()) / stride + 1;

	Matrix C(out_rows, out_cols);
	for (size_t i = 0; i < out_rows; i++) {
		for (size_t j = 0; j < out_cols; j++) {
			double sum = 0;
			for (size_t kIdx_r = 0; kIdx_r < kernel.rows(); kIdx_r++) {
				for (size_t kIdx_c = 0; kIdx_c < kernel.cols(); kIdx_c++) {
					int inputRow = i * stride + kIdx_r - padding;
					int inputCol = j * stride + kIdx_c - padding;
					if (inputRow >= 0 && inputRow < input.rows() && inputCol >= 0 && inputCol < input.cols()) {
						sum += kernel(kIdx_r, kIdx_c) * input(inputRow, inputCol);
					}
				}
			}
			C(i, j) = sum;
		}
	}

	return C;
}

Matrix Matrix::rotate180() const {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = (_rows - 1 - i) * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + (_cols - 1 - j)];
	}
	return C;
}

Matrix Matrix::dilate(const int stride) const {

	if (stride == 1) return *this;

	size_t new_rows = (_rows - 1) * stride + 1;
	size_t new_cols = (_cols - 1) * stride + 1;

	Matrix C(new_rows, new_cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;

		for (size_t j = 0; j < _cols; j++)
			C(i * stride, j * stride) = _matrix[row_offset + j];
	}
	return C;
}

double Matrix::norm() const {

	double norm = 0;
	for (size_t i = 0; i < _rows - 1; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			norm += _matrix[row_offset + j];
	}
	return norm / _rows / _cols;
}