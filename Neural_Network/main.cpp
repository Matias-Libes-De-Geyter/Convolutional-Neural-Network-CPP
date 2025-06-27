#include "Trainer/TrainerClassifier.h"
#include <fstream>
#pragma GCC diagnostic ignored "-Wnarrowing"

// Hyperparameters declaration
hyperparameters current_hyperparameters = {
	number_classes : 10,
	hidden_layers_sizes : { 256, 128 },
	epochs : 0,
	mini_batch_size : 64,
	learning_rate : 0.001,
	dropout_rate : 0,
	learn : false,
	test : true,
	store_weights : false,
	store_output_data : false,
	filters : { 32, 64 },
	kernel_size : 3,
	padding : 0,
	stride : 2,
	img_size : 28,
	database : "numbers"
};

int main() {
	// User interface?
	bool user_interface = false;
	if (user_interface) {
		// Ask for database type
		print("What's your plan ? MNIST's Numbers or Fashion ? (n/f)"); char z; std::cin >> z;
		(z == 'f') ? current_hyperparameters.database = "fashion" : current_hyperparameters.database = "numbers";
		
		// Ask for model training
		print("Train ? (y/n)"); char a; std::cin >> a;
		if (a == 'y') {
			current_hyperparameters.learn = true;

			// Ask for model storing
			print("Store model ? (y/n)"); char b; std::cin >> b;
			if (b == 'y') current_hyperparameters.store_weights = true;
		}

		// Ask for model testing at the end
		print("Test ? (y/n)"); char b; std::cin >> b;
		if (b == 'y') current_hyperparameters.test = true;
	}

	// MLP init
	CNN NN(current_hyperparameters);

	// Training
	if (current_hyperparameters.learn) {
		// Train and store weights
		train(NN, current_hyperparameters);
		if (current_hyperparameters.store_weights)
			NN.saveWeights();
		print("Weights & Kernels stored !");
	}
	else {
		// Load weights
		NN.loadWeights();
		print("Weights & Kernels loaded !");
	}
	// Testing
	if(current_hyperparameters.test) test(NN, current_hyperparameters);

}


