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
#include "include/environments/atari_prediction_environment.h"
#include <unordered_map>
#include "include/stdout.h"
#include "include/evaluators/predictionEvaluator.h"

// Data preprocessing to find the salient features
std::vector<float> std_of_ram_data(128, 0);
std::vector<float> mean_of_ram_data(128, 0);
std::vector<float> max_of_ram_data(128, -1);
std::vector<float> min_of_ram_data(128, 2);

std::vector<float> normalized_byte_data(128, 0);

std::unordered_map<int, int> feature_indices;


bool cmp(int& a, int& b)
{
	return std_of_ram_data[a] > std_of_ram_data[b];
}

float getNormalizedByteData(unsigned char b, int index)
{
	int in = int(b);
	float normalized_byte_data = float(in) / 256.0f;
	normalized_byte_data = (normalized_byte_data - min_of_ram_data[index]) / (max_of_ram_data[index] -
		min_of_ram_data[index]);
	normalized_byte_data = (normalized_byte_data - mean_of_ram_data[index]) / std_of_ram_data[index];
	return normalized_byte_data;
}


int main(int argc, char* argv[])
{
	Experiment* my_experiment = new ExperimentJSON(argc, argv);
	if (my_experiment->get_int_param("sum_features") && !my_experiment->get_int_param("n2_decorrelate"))
		exit(1);

	Logger* logger;
	logger = new MongoDBLogger("mongodb://admin:***@34.95.16.129:27017/", "Test1",
	                           "Test2");


	nlohmann::json j;
	for (auto const& x : my_experiment->args_for_run)
	{
		j[x.first] = x.second;
	}

	PredictionEvaluator evaluator(my_experiment->get_float_param("gamma"));
	AtariPredictionEnvironment env(my_experiment->get_string_param("game"), my_experiment->get_float_param("gamma"),
	                               "pretrained");
	bool to_print = true;

	// Compute range of all features
	for (int i = 0; i < 100000; i++)
	{
		env.step();
		for (int j = 0; j < env.my_env->getRAM().size(); j++)
		{
			unsigned char b = env.my_env->getRAM().get(j);
			int in = int(b);
			normalized_byte_data[j] = float(in) / 256.0f;
			// std::cout << normalized_byte_data[j] << " " << j << std::endl;
			if (max_of_ram_data[j] < normalized_byte_data[j])
				max_of_ram_data[j] = normalized_byte_data[j];
			if (min_of_ram_data[j] > normalized_byte_data[j])
				min_of_ram_data[j] = normalized_byte_data[j];
		}
	}

	// Compute variance of normalized ram features
	for (int i = 0; i < 100000; i++)
	{
		env.step();
		for (int j = 0; j < env.my_env->getRAM().size(); j++)
		{
			unsigned char b = env.my_env->getRAM().get(j);
			int in = int(b);
			normalized_byte_data[j] = float(in) / 256.0f;
			normalized_byte_data[j] = (normalized_byte_data[j] - min_of_ram_data[j]) / (max_of_ram_data[j] -
				min_of_ram_data[j]);
			// check if nan or inf and replace with 0
			if (normalized_byte_data[j] != normalized_byte_data[j] || normalized_byte_data[j] == std::numeric_limits<
				float>::infinity())
			{
				normalized_byte_data[j] = 0;
			}
		}
		for (int i = 0; i < 128; i++)
		{
			mean_of_ram_data[i] = 0.9995f * mean_of_ram_data[i] + 0.0005f * normalized_byte_data[i];
			std_of_ram_data[i] = 0.9995f * std_of_ram_data[i] + 0.0005f * (normalized_byte_data[i] - mean_of_ram_data[
				i]) * (normalized_byte_data[i] - mean_of_ram_data[i]);
		}
	}


	// Get indices of ram positions sorted by std
	std::vector<int> indices(128);
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), cmp);

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

	std::mt19937 mt(my_experiment->get_int_param("seed"));
	int total_inputs = my_experiment->get_int_param("n_inputs") + my_experiment->get_int_param("n_distractors");
	SyncedNeuron::neuron_id_generator = 0;
	auto learning_network = SingleLayerNetwork(my_experiment->get_float_param("step_size"),
	                                           my_experiment->get_int_param("seed"),
	                                           total_inputs,
	                                           my_experiment->get_int_param("n_learner_features"),
	                                           false);

	auto input_sampler = uniform_random(my_experiment->get_int_param("seed"), -10, 10);

	double running_error = 0.0;
	std::vector<std::pair<std::pair<float, float>, std::string>> graphs;
	int counter = 0;


	if (to_print) std::cout << std::fixed;
	if (to_print) std::cout << ansii::clear_screen();
	if (to_print) ansii::print_box(4, 4, 60, 100);

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



		env.step();
		std::vector<float> input;
		for (int i = 0; i < my_experiment->get_int_param("n_inputs"); i++)
		{
			float val = getNormalizedByteData(env.my_env->getRAM().get(indices[i]), indices[i]);
			input.push_back(val);
		}


		learning_network.forward(input);
		learning_network.calculate_all_correlations();
		// running_error += (target - pred) * (target - pred);

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

		float delta = env.get_reward() + my_experiment->get_float_param("gamma")*learning_network.predictions - learning_network.v_old;
		if(delta != 0)
		{
			if(to_print) ansii::printKeyValue(50, 6, "Delta", std::to_string(delta));
		}
		if(to_print) ansii::printKeyValue(50, 5, "Pred", std::to_string(learning_network.predictions));
		evaluator.addPredictionAndReward(learning_network.predictions, env.get_reward());

		// if (my_experiment->get_int_param("freeze_weights"))
		// 	learning_network.update_parameters_only_prediction_RMSProp(delta);
		// else
		learning_network.update_parameters_only_prediction(delta);
		learning_network.decayGradients(my_experiment->get_float_param("gamma")*my_experiment->get_float_param("lambda"));
		learning_network.backward();


		// learning_network.update_parameters(delta);



		if (step % 1000 == 1)
		{
			if(to_print) ansii::printKeyValue(10, 9, "Return error", std::to_string(evaluator.getMSEOverAllExperience()));
			if (to_print) ansii::printKeyValue(10, 5, "Average Error", std::to_string(running_error / step));
			if (to_print) ansii::printKeyValue(10, 6, "Step", std::to_string(step));
			if(to_print) ansii::printKeyValue(10, 7, "Correlated Features", std::to_string(learning_network.count_highly_correlated_features()));
			if(to_print) ansii::printKeyValue(10, 8, "Mature Features", std::to_string(learning_network.count_mature_features()));
		}
		if (step % 5000 == 1)
		{
			if(evaluator.getAge() > 3000)
			{
				auto preds = evaluator.getPredictions(evaluator.getAge() - 2000, evaluator.getAge() - 1950);
				auto returns = evaluator.getReturns(evaluator.getAge() - 2000, evaluator.getAge() - 1950);
				for(int i = 0; i < 50; i ++)
				{
					if (to_print) ansii::printKeyValue(10, 10 + i, "Pred", std::to_string(preds[i].second));
					if (to_print) ansii::printKeyValue(50, 10 + i, "Return", std::to_string(returns[i].second));
				}
			}
			error_data.push_back(std::make_pair(step, running_error / step));
			n_correlated_data.push_back(std::make_pair(step, learning_network.count_highly_correlated_features()));
			n_mature_data.push_back(std::make_pair(step, learning_network.count_mature_features()));
		}
		// learning_network.zero_grad();
		if (to_print) std::cout << ansii::move_cursor(0, 101) << std::flush;
	}
	// std::cout << "Experiment finished" << std::endl;

	j["error_data"] = error_data;
	j["n_correlated_data"] = n_correlated_data;
	j["n_mature_data"] = n_mature_data;
	j["Final error"] = running_error / my_experiment->get_int_param("steps");
	j["Final n_correlated"] = learning_network.count_highly_correlated_features();
	j["Final n_mature"] = learning_network.count_mature_features();
	j["real_correlation"] = real_correlation;
	j["estimated_correlation"] = estimated_correlation;
	// std::cout << j.dump() << std::endl;
	logger->log(j.dump());
}
