#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <memory>

#include <torch/script.h>
#include <torch/data.h>
#include <torch/nn.h>

#include "../include/utils.h"
#include "../include/nn/networks/pretrained_dense_network.h"
#include "../include/nn/utils.h"



int main(int argc, char *argv[]){
	torch::jit::script::Module trained_model;
	try {
		std::cout << "loading the torch model from: \t" << "trained_models/mnist_small.pt" << std::endl;
		trained_model = torch::jit::load("trained_models/mnist_small.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	PretrainedDenseNetwork network = PretrainedDenseNetwork(trained_model,
	                                                        0,
	                                                        1,
	                                                        14*14,
	                                                        0.0,
                                                            0.0,
                                                            0,
                                                            10000,
                                                            0,
                                                            0.999);


	std::cout << "Network confing\n";
	std::cout << "No\tSize\tSynapses\tOutput\n";
	for(int layer_no = 0; layer_no < network.all_neuron_layers.size(); layer_no++)
		std::cout <<  layer_no << "\t" << network.all_neuron_layers[layer_no].size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;
	std::cout <<  1000 << "\t" << network.output_neurons.size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;

	// preprocess dataset
	auto train_dataset =
		torch::data::datasets::MNIST("data/")
		.map(torch::data::transforms::Normalize<>(0.1307, 0.2801))
		.map(torch::data::transforms::Stack<>());
	int total_training_items = 60000;
	std::vector<std::vector<float> > images;
	std::vector<std::vector<float> > targets;

	for(int counter = 0; counter < total_training_items; counter++) {
		std::vector<float> x_temp;
		auto x = torch::nn::functional::interpolate(train_dataset.get_batch(counter).data,
		                                            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({14,14})).mode(torch::kBilinear).align_corners(false));
		auto x_vec = x.reshape({14*14});
		for (int i = 0; i < 14*14; i++)
			x_temp.push_back(x_vec.index({i}).item<float>());
		images.push_back(x_temp);

		std::vector<float> y_vec;
		y_vec.push_back(train_dataset.get_batch(counter).target.item<float>());
		targets.push_back(y_vec);
	}

	// pred on mnist input
	auto x = images[0];
	float y_index = targets[0][0];
	std::vector<float> y(10);
	y[y_index] = 1;

	std::cout << "input: ";
	print_vector(x);
	network.forward(x);
	auto prediction = network.read_output_values();
	std::cout << "pred: ";
	print_vector(prediction);


	// pred on vector of ones
	std::cout << "\n\n\n\n" << std::endl;
	std::vector<float> sample_of_ones(14*14, 1.0);
	std::cout << "input: ";
	print_vector(sample_of_ones);
	network.forward(sample_of_ones);
	prediction = network.read_output_values();
	std::cout << "pred: ";
	print_vector(prediction);

//  std::cout << "\n\n\n\nvalues: " << std::endl;
//  int layer_no = 0;
//  for (const auto& l : network.all_neuron_layers){
//    std::vector<float> values;
//    std::cout << "layer: " << layer_no++ << std::endl;
//    for (const auto& n : l)
//      values.push_back(n->value);
//    print_vector(values);
//  }


	std::cout << "\n\n\nweights: " << std::endl;
	int layer_no = 0;
	for (const auto& l : network.all_neuron_layers) {
		std::cout << "layer: " << layer_no++ << std::endl;
		for (const auto& n : l) {
			std::vector<float> weights;
			for (const auto& s : n->incoming_synapses)
				weights.push_back(s->weight);
			print_vector(weights);
		}
		std::cout << "\n\n" << std::endl;
	}
	std::cout << "layer: " << 1000 << std::endl;
	for (const auto& n : network.output_neurons) {
		std::vector<float> weights;
		for (const auto& s : n->incoming_synapses)
			weights.push_back(s->weight);
		print_vector(weights);
	}
}
