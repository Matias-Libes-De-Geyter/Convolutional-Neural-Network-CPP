#include "TrainerClassifier.h"


// Train function
void train(CNN& NN, const hyperparameters& hyper) {
	// Get images & labels from database
	dmatrix images;
	dvector labels;
	std::string trainImagesFile = "database/MNIST/train-images.idx3-ubyte";
	std::string trainLabelsFile = "database/MNIST/train-labels.idx1-ubyte";
	if (hyper.database == "fashion") {
		trainImagesFile = "database/MNIST_FASHION/train-images.idx3-ubyte";
		trainLabelsFile = "database/MNIST_FASHION/train-labels.idx1-ubyte";
	}
	readMNIST(trainImagesFile, trainLabelsFile, images, labels);
	Matrix labels_hotOnes = hotOne(labels, 10);
	print("Database finished loading.");

	// Compute epochs
	int epochs = (hyper.epochs == 0 ? (60000.f / hyper.mini_batch_size) : hyper.epochs);
	print("Number of epochs: ", epochs); print("");

	// Training loop
	for (int i = 0; i < epochs; i++) {
		// Get mini_batch_size images
		dmatrix x_train(&images[hyper.mini_batch_size * i], &images[hyper.mini_batch_size * (i + 1)]);
		dmatrix y_test(&labels_hotOnes[hyper.mini_batch_size * i], &labels_hotOnes[hyper.mini_batch_size * (i + 1)]);

		// Flatten them
		std::vector<Matrix> input_images;
		for (int i = 0; i < hyper.mini_batch_size; i++)
			input_images.push_back(flattenToMatrix(x_train[i], hyper.img_size, hyper.img_size));

		// Put them through the CNN, make it learn from them and get the loss
		Matrix y_train = NN.forward(input_images);
		double loss = NN.backwards(input_images, y_test);
	}
}

// Test function
void test(CNN& NN, const hyperparameters& hyper) {

	// Get images & labels from database
	dmatrix imagesTests;
	dvector labelsTests;
	std::string testImagesFile = "database/MNIST/t10k-images.idx3-ubyte";
	std::string testLabelsFile = "database/MNIST/t10k-labels.idx1-ubyte";
	if (hyper.database == "fashion") {
		testImagesFile = "database/MNIST_FASHION/t10k-images.idx3-ubyte";
		testLabelsFile = "database/MNIST_FASHION/t10k-labels.idx1-ubyte";
	}
	readMNIST(testImagesFile, testLabelsFile, imagesTests, labelsTests);
	dmatrix labelsTestsHot = hotOne(labelsTests, 10);
	print("Test database finished loading.");

	// Flatten images
	std::vector<Matrix> input_images;
	for (int i = 0; i < imagesTests.size(); i++)
		input_images.push_back(flattenToMatrix(imagesTests[i], hyper.img_size, hyper.img_size));

	// Put them through the CNN
	dmatrix y_train_certain = NN.forward(input_images).setMaxToOne();

	// Compute the accuracy
	int acc = 0;
	for (int i = 0; i < y_train_certain.size(); i++)
		if (labelsTestsHot[i] == y_train_certain[i])
			acc += 1;
	print("");
	print("The accuracy is ", 100.f * acc / y_train_certain.size(), " %");

}