#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "include/nn/networks/single_layer_network.h"
#include "include/nn/synced_neuron.h"
#include "include/logger/logger.h"
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/exception/exception.hpp>
#include "include/environments/atari_prediction_environment.h"
#include <unordered_map>

std::vector<float> std_of_ram_data(128, 0);
std::vector<float> mean_of_ram_data(128, 0);

std::unordered_map<int, int> feature_indices;
int main(int argc, char* argv[])
{
	Experiment* my_experiment = new ExperimentJSON(argc, argv);
	if (my_experiment->get_int_param("sum_features") && !my_experiment->get_int_param("n2_decorrelate"))
		exit(1);

	Logger* logger;
	logger = new MongoDBLogger("mongodb://admin:rlc20251234@34.95.16.129:27017/", "Test1",
	                           "Test2");


	nlohmann::json j;
	for (auto const& x : my_experiment->args_for_run)
	{
		j[x.first] = x.second;
	}

	AtariPredictionEnvironment env("Pong", 0.99, "pretrained");

	std::vector<std::pair<int, float>> error_data;
	std::vector<std::pair<int, float>> n_correlated_data;
	std::vector<std::pair<int, float>> n_mature_data;

	std::vector<std::pair<int, std::pair<int, float>>> real_correlation;
	std::vector<std::pair<int, std::pair<int, float>>> estimated_correlation;


	std::vector<std::string> weight_col_names{"run", "step"};
	std::vector<std::string> weight_col_types{"int", "int"};
	for (int i = 0; i < my_experiment->get_int_param("n_learner_features"); i++)
	{
		weight_col_names.push_back("f" + std::to_string(i));
		weight_col_types.push_back("real");
	}

	for (int i = 0; i < my_experiment->get_int_param("n_learner_features"); i++)
	{
		weight_col_names.push_back("age" + std::to_string(i));
		weight_col_types.push_back("int");
	}
	// Metric weight_metric = Metric(my_experiment->database_name, "weights_table",
	//                               weight_col_names,
	//                               weight_col_types,
	//                               std::vector < std::string > {"run", "step"});

	std::cout << "Program started \n";

	std::mt19937 mt(my_experiment->get_int_param("seed"));
	int total_inputs = my_experiment->get_int_param("n_inputs") + my_experiment->get_int_param("n_distractors");
	auto target_network = SingleLayerNetwork(0.0,
	                                         my_experiment->get_int_param("seed") + 1000,
	                                         my_experiment->get_int_param("n_inputs"),
	                                         my_experiment->get_int_param("n_target_features"),
	                                         true);
	SyncedNeuron::neuron_id_generator = 0;
	auto learning_network = SingleLayerNetwork(my_experiment->get_float_param("step_size"),
	                                           my_experiment->get_int_param("seed"),
	                                           total_inputs,
	                                           my_experiment->get_int_param("n_learner_features"),
	                                           false);

	auto input_sampler = uniform_random(my_experiment->get_int_param("seed"), -10, 10);

	float running_error = 0.05;
	std::vector<std::pair<std::pair<float, float>, std::string>> graphs;
	int counter = 0;

	for (int step = 0; step < my_experiment->get_int_param("steps"); step++)
	{
		if (step % my_experiment->get_int_param("replace_every") == 1)
		{
			if (my_experiment->get_int_param("n2_decorrelate"))
				graphs = learning_network.replace_features_n2_decorrelator_v3(
					my_experiment->get_float_param("replace_perc"),
					bool(my_experiment->get_int_param("sum_features")),
					my_experiment->get_float_param("decorrelate_perc"),
					bool(my_experiment->get_int_param("use_generate_and_test")));
			else if (my_experiment->get_int_param("random_decorrelate") || my_experiment->get_int_param(
				"random_thresh_decorrelate"))
				graphs = learning_network.replace_features_random_decorrelator_v3(
					my_experiment->get_float_param("replace_perc"),
					bool(my_experiment->get_int_param("sum_features")),
					my_experiment->get_float_param("decorrelate_perc"));
			else if (my_experiment->get_int_param("random_replacement"))
				learning_network.replace_features_randomly(my_experiment->get_float_param("replace_perc"));
			else if (bool(my_experiment->get_int_param("use_generate_and_test")))
				learning_network.replace_features(my_experiment->get_float_param("replace_perc"));

			for (const auto& graph : graphs)
			{
				real_correlation.push_back(std::make_pair(step, std::make_pair(counter, graph.first.first)));
				estimated_correlation.push_back(std::make_pair(step, std::make_pair(counter, graph.first.second)));
				counter++;
			}
		}


		auto input = input_sampler.get_random_vector(total_inputs);
		std::vector<int> active_indices;
		std::vector<float> active_indices_values;
		env.step();
		auto byte_data =  env.my_env->getRAM().array();
		std::vector<float> normalized_byte_data{128, 0};
		for(int i = 0; i < 128; i++)
		{
			normalized_byte_data[i] = float(int(byte_data[i]))/256.0f;
		}
		std::cout << "Step " << step << std::endl;
		for(int i = 0; i < 128; i++)
		{
			mean_of_ram_data[i] = 0.999f * mean_of_ram_data[i] + 0.001f * normalized_byte_data[i];
			std_of_ram_data[i] = 0.999f* std_of_ram_data[i] + 0.001f * (normalized_byte_data[i] - mean_of_ram_data[i]) * (normalized_byte_data[i] - mean_of_ram_data[i]);
			if(std_of_ram_data[i] > 0.01)
			{
				if(feature_indices.find(i) == feature_indices.end())
				{
					feature_indices[i] = feature_indices.size();
				}
				active_indices.push_back(feature_indices[i]);
				std::cout << i << "\t" <<  std_of_ram_data[i] << std::endl;
				active_indices_values.push_back((normalized_byte_data[i] - mean_of_ram_data[i])/std_of_ram_data[i]);
			}
		}
		std::cout << "Total features = " << feature_indices.size() << std::endl;
		// std::vector<float> input(5, 0);

		float pred = learning_network.forward(input);
		float target = target_network.forward(input);
		float error = target - pred;


		learning_network.calculate_all_correlations();
		running_error = 0.995 * running_error + 0.005 * (target - pred) * (target - pred);

		if (my_experiment->get_int_param("random_decorrelate"))
		{
			if ((my_experiment->get_int_param("age_restriction") && step > 25000) || !my_experiment->get_int_param(
				"age_restriction"))
			{
				if (step % my_experiment->get_int_param("min_estimation_period") == 1)
					//update the random corr selections
					learning_network.update_random_correlation_selections(
						bool(my_experiment->get_int_param("age_restriction")),
						my_experiment->get_float_param("perc_of_total_pairs_to_estimate"));
				learning_network.calculate_random_correlations(my_experiment->get_int_param("min_estimation_period"));
				// update the random corr values
			}
		}

		// same random decorrelator as above but sampling based on correlations based on a single sample
		if (my_experiment->get_int_param("random_thresh_decorrelate"))
		{
			if ((my_experiment->get_int_param("age_restriction") && step > 25000) || !my_experiment->get_int_param(
				"age_restriction"))
			{
				if (step % my_experiment->get_int_param("min_estimation_period") == 1)
					//update the random corr selections
					learning_network.update_random_correlation_selections_using_thresh(
						bool(my_experiment->get_int_param("age_restriction")));
				learning_network.calculate_random_correlations(my_experiment->get_int_param("min_estimation_period"));
				// update the random corr values
			}
		}

		learning_network.backward();
		learning_network.update_parameters(error);
		//learning_network.update_parameters_only_prediction(error);
		//learning_network.update_parameters_only_prediction(error,
		//                                                   my_experiment->get_float_param("l2_lambda"),
		//                                                   my_experiment->get_float_param("l1_lambda"));
		if (my_experiment->get_int_param("freeze_weights"))
			learning_network.update_parameters_only_prediction_RMSProp(error);
		else
			learning_network.update_parameters(error);

		if (step % 5000 == 1)
		{
			// || step%5000 == 4999){
			error_data.push_back(std::make_pair(step, running_error));
			n_correlated_data.push_back(std::make_pair(step, learning_network.count_highly_correlated_features()));
			n_mature_data.push_back(std::make_pair(step, learning_network.count_mature_features()));

			// error_metric.record_value(cur_error);
			std::cout << "\nstep:" << step << std::endl;
			//print_vector(input);
			//print_vector(learning_network.get_prediction_weights());
			//print_vector(learning_network.get_feature_utilities());
			//print_vector(learning_network.get_prediction_gradients());
			std::cout << "target: " << target << " pred: " << pred << std::endl;
			std::cout << "running err: " << running_error << std::endl;
			//learning_network.print_all_correlations();
			//learning_network.print_all_statistics();
			std::cout << "count unremovable correlated features: " << learning_network.
				count_highly_correlated_features() << std::endl;
			std::cout << "count mature features" << learning_network.count_mature_features() << std::endl;
		}
		learning_network.zero_grad();
		if (step % 100000 == 1)
		{
			// error_metric.commit_values();
			// correlation_metric.commit_values();
			// weight_metric.commit_values();
		}
	}
	std::cout << "Experiment finished" << std::endl;

	j["error_data"] = error_data;
	j["n_correlated_data"] = n_correlated_data;
	j["n_mature_data"] = n_mature_data;
	j["Final error"] = running_error;
	j["Final n_correlated"] = learning_network.count_highly_correlated_features();
	j["Final n_mature"] = learning_network.count_mature_features();
	j["real_correlation"] = real_correlation;
	j["estimated_correlation"] = estimated_correlation;
	std::cout << j.dump() << std::endl;
	logger->log(j.dump());
}
