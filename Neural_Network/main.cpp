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
	early_stopping : false,
	patience : 150,
	learn : false,
	test : true,
	store_weights : false,
	filters : { 32, 64 },
	kernel_size : 3,
	padding : 0,
	stride : 2,
	img_size : 28,
	database : "numbers"
};

// Results
//// av. très lent. Enlevé les normalisations. Bien plus rapide:
//////// 64mb, { 32, 64 }cl, (3, 0, 2), { 128 }fl ->				20e = 0.76loss (peu stable également)
//////// 64mb, { 32, 64 }cl, (3, 0, 2), { 256, 512 }fl -> 10e = 1loss |	20e = 0.6loss	(fluctue beaucoup ! Avec normalisation, fluctuait très très peu. Mais bcp plus lent.)
// //////// 64mb, { 32, 64 }cl, (3, 0, 2), { 256, 512 }fl -> 97.12% sur les nombres.
//////// 64mb, { 16, 32, 64 }cl, (5, 0, 1), { 256, 512 }fl ->			20e = 1.5loss


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


