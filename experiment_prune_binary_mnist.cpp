#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <string>
#include <random>

#include <torch/script.h>
#include <torch/data.h>
#include <torch/nn.h>

#include "include/utils.h"
#include "include/nn/networks/pretrained_dense_network.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"


int main(int argc, char *argv[]){
	auto start = std::chrono::steady_clock::now();
	double running_error = 6;
	double accuracy = 0.1;
	float alpha = 0.999;
	Experiment *exp = new ExperimentJSON(argc, argv);
	std::mt19937 mt(exp->get_int_param("seed"));
	std::vector<std::vector<std::string> > error_logger;
	std::vector<std::vector<std::string> > state_logger;

	Metric error_metric = Metric(exp->database_name, "error_table",
	                             std::vector < std::string > {"step", "run", "error", "accuracy", "n_params", "param_schedule"},
	                             std::vector < std::string > {"int", "int", "real", "real", "int", "int"},
	                             std::vector < std::string > {"step", "run"});
	Metric state_metric = Metric(exp->database_name, "state_metric",
	                             std::vector < std::string > {"step", "run", "layer_no", "nparams"},
	                             std::vector < std::string > {"int", "int", "int", "int"},
	                             std::vector < std::string > {"step", "run", "layer_no"});
	Metric run_state_metric = Metric(exp->database_name, "run_state",
	                                 std::vector < std::string > {"run", "error", "accuracy", "state", "run_time"},
	                                 std::vector < std::string > {"int", "real", "real", "VARCHAR(40)", "VARCHAR(60)"},
	                                 std::vector < std::string > {"run"});


	torch::jit::script::Module trained_model;
	try {
		// Deserialize the trained pytorch model weights
		std::cout << "loading the torch model from: \t" << exp->get_string_param("trained_model_path") << std::endl;
		trained_model = torch::jit::load(exp->get_string_param("trained_model_path") + std::to_string(exp->get_int_param("seed")) + ".pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	PretrainedDenseNetwork net = PretrainedDenseNetwork(exp->get_int_param("seed"),
	                                                    exp->get_int_param("min_synapses_to_keep"),
	                                                    exp->get_int_param("prune_interval"),
	                                                    exp->get_int_param("start_pruning_at"),
	                                                    exp->get_float_param("trace_decay_rate"));
	if (exp->get_int_param("linear_only"))
		net.load_linear_network(trained_model,
		                        exp->get_float_param("step_size"),
		                        14*14,
		                        exp->get_float_param("utility_to_keep"));
	else
		net.load_relu_network(trained_model,
		                      exp->get_float_param("step_size"),
		                      14*14,
		                      exp->get_float_param("utility_to_keep"));


	//load the mnist dataset and normalize it
	auto train_dataset =
		torch::data::datasets::MNIST("data/")
		.map(torch::data::transforms::Normalize<>(0.1307, 0.2801))
		.map(torch::data::transforms::Stack<>());
	int total_training_items = exp->get_int_param("training_points");
	std::vector<std::vector<float> > images;
	std::vector<float> targets;

	// resize all images to 14x14 and convert the labels to binary even/odd labels
	for(int counter = 0; counter < total_training_items; counter++) {
		std::vector<float> x_temp;
		auto x = torch::nn::functional::interpolate(train_dataset.get_batch(counter).data,
		                                            torch::nn::functional::InterpolateFuncOptions()
		                                            .size(std::vector<int64_t>({14,14}))
		                                            .mode(torch::kBilinear).align_corners(false));
		auto x_vec = x.reshape({14*14});
		for (int i = 0; i < 14*14; i++)
			x_temp.push_back(x_vec.index({i}).item<float>());
		images.push_back(x_temp);

		if (int(train_dataset.get_batch(counter).target.item<float>()) % 2 == 0)
			targets.push_back(0.0);
		else
			targets.push_back(1.0);
	}


	int total_data_points = exp->get_int_param("training_points");
	bool update_weights = true;
	if (exp->get_float_param("step_size") == 0)
		update_weights = false;

	std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);

	for (int step = 0; step < exp->get_int_param("steps"); step++) {
		int index = index_sampler(mt);
		auto x = images[index];
		std::vector y = {targets[index]};

		// forward pass
		net.forward(x);
		auto prediction = net.read_output_values();

		float error = 0;
		for(int j = 0; j<prediction.size(); j++) {
			error += (prediction[j]-y[j])*(prediction[j]-y[j]);
		}
		running_error = running_error * alpha + (1-alpha) * std::sqrt(error);
		if( (prediction[0] > 0.5 && y[0] == 1) || (prediction[0] <=0.5 && y[0] == 0) )
			accuracy = accuracy*alpha + (1-alpha);
		else
			accuracy *= alpha;

		// compute gradients
		if (true || update_weights)
			net.backward(y);

		net.update_utility_estimates(exp->get_string_param("pruner_type"),
		                             x,
		                             prediction,
		                             exp->get_int_param("dropout_estimator_iterations"),
		                             exp->get_float_param("dropout_perc"));
		net.prune_weights(exp->get_string_param("pruner_type"));

		// update weights
		if (update_weights)
			net.update_weights();

		if (step % 10000 == 0) {
			std::vector<std::string> error_vec;
			error_vec.push_back(std::to_string(step));
			error_vec.push_back(std::to_string(exp->get_int_param("run")));
			error_vec.push_back(std::to_string(running_error));
			error_vec.push_back(std::to_string(accuracy));
			error_vec.push_back(std::to_string(net.all_synapses.size()));
			error_vec.push_back(std::to_string(net.get_current_synapse_schedule()));
			error_logger.push_back(error_vec);
		}

		if (step % 500000 == 0) {
			std::cout << "Step " << step << std::endl;
			std::cout << "Network confing\n";
			std::cout << "No\tSize\tSynapses\tOutput\n";


			for(int layer_no = 0; layer_no < net.all_neuron_layers.size(); layer_no++) {
				int n_layer_synapses = 0;
				for (auto n : net.all_neuron_layers[layer_no])
					n_layer_synapses += int(n->incoming_synapses.size());
				std::vector<std::string> state_vec;
				state_vec.push_back(std::to_string(step));
				state_vec.push_back(std::to_string(exp->get_int_param("run")));
				state_vec.push_back(std::to_string(layer_no));
				state_vec.push_back(std::to_string(n_layer_synapses));
				state_logger.push_back(state_vec);
				std::cout <<  layer_no << "\t" << net.all_neuron_layers[layer_no].size() << "\t"<< n_layer_synapses;
				std::cout << "/" << net.all_synapses.size() << "\t\t" << net.output_synapses.size() <<  std::endl;
			}
			int n_layer_synapses = 0;
			for (auto n : net.output_neurons)
				n_layer_synapses += int(n->incoming_synapses.size());
			std::vector<std::string> state_vec;
			state_vec.push_back(std::to_string(step));
			state_vec.push_back(std::to_string(exp->get_int_param("run")));
			state_vec.push_back(std::to_string(1000));
			state_vec.push_back(std::to_string(n_layer_synapses));
			state_logger.push_back(state_vec);
			std::cout <<  1000 << "\t" << net.output_neurons.size() << "\t" << n_layer_synapses;
			std::cout << "/" << net.all_synapses.size() << "\t\t" << net.output_synapses.size() <<  std::endl;

			std::cout << "Target: ";
			print_vector(y);
			std::cout << "Prediction: ";
			print_vector(prediction);
			std::cout << "Running accuracy: " << accuracy << std::endl;
			std::cout << "Running error = " << running_error << std::endl;
		}


		if(step % 100000 == 0) {
			std::cout << error_logger.size() << std::endl;
			error_metric.add_values(error_logger);
			state_metric.add_values(state_logger);
			error_logger.clear();
			state_logger.clear();
		}
	}
	error_metric.add_values(error_logger);
	state_metric.add_values(state_logger);
	error_logger.clear();
	state_logger.clear();

	// track run state and run time
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::to_string(std::chrono::duration_cast<std::chrono::minutes>(end - start).count());
	std::vector<std::string> state_data;
	state_data.push_back(std::to_string(exp->get_int_param("run")));
	state_data.push_back(std::to_string(running_error));
	state_data.push_back(std::to_string(accuracy));
	state_data.push_back("finished");
	state_data.push_back(elapsed);
	run_state_metric.add_value(state_data);
}
