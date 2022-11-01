#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "include/nn/networks/single_layer_network.h"
#include "include/nn/synced_neuron.h"


int main(int argc, char *argv[]) {

	Experiment *my_experiment = new ExperimentJSON(argc, argv);
	if (my_experiment->get_int_param("sum_features") && !my_experiment->get_int_param("n2_decorrelate"))
		exit(1);

	Metric error_metric = Metric(my_experiment->database_name, "error_table",
	                             std::vector < std::string > {"run", "step", "error", "n_correlated", "n_mature"},
	                             std::vector < std::string > {"int", "int", "real", "int", "int"},
	                             std::vector < std::string > {"run", "step"});

	Metric correlation_metric = Metric(my_experiment->database_name, "correlated_graphs_table",
	                                   std::vector < std::string > {"run", "step", "id",  "real_correlation", "estimated_correlation", "graph"},
	                                   std::vector < std::string > {"int", "int", "int", "real", "real", "varchar(10000)"},
	                                   std::vector < std::string > {"run", "step", "id"});

	Metric summary_metric = Metric(my_experiment->database_name, "summary_table",
	                               std::vector < std::string > {"run", "final_error", "final_n_correlated", "final_n_mature"},
	                               std::vector < std::string > {"int", "real", "int", "int"},
	                               std::vector < std::string > {"run"});

	std::vector<std::string> weight_col_names{ "run", "step"};
	std::vector<std::string> weight_col_types{ "int", "int"};
	for (int i = 0; i < my_experiment->get_int_param("n_learner_features"); i++) {
		weight_col_names.push_back("f" + std::to_string(i));
		weight_col_types.push_back("real");
	}

	for (int i = 0; i < my_experiment->get_int_param("n_learner_features"); i++) {
		weight_col_names.push_back("age" + std::to_string(i));
		weight_col_types.push_back("int");
	}
	Metric weight_metric = Metric(my_experiment->database_name, "weights_table",
	                              weight_col_names,
	                              weight_col_types,
	                              std::vector < std::string > {"run", "step"});

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
	std::vector<std::pair<std::pair<float, float>, std::string> > graphs;
	int counter = 0;

	for (int step = 0; step < my_experiment->get_int_param("steps"); step++) {
		if (step % my_experiment->get_int_param("replace_every") == 1) {
			if (my_experiment->get_int_param("n2_decorrelate"))
				graphs = learning_network.replace_features_n2_decorrelator_v3(my_experiment->get_float_param("replace_perc"),
				                                                              bool(my_experiment->get_int_param("sum_features")),
				                                                              my_experiment->get_float_param("decorrelate_perc"));
			else if (my_experiment->get_int_param("random_decorrelate") || my_experiment->get_int_param("random_thresh_decorrelate"))
				graphs = learning_network.replace_features_random_decorrelator_v3(my_experiment->get_float_param("replace_perc"),
				                                                                  bool(my_experiment->get_int_param("sum_features")),
				                                                                  my_experiment->get_float_param("decorrelate_perc"));
			else if (my_experiment->get_int_param("random_replacement"))
				learning_network.replace_features_randomly(my_experiment->get_float_param("replace_perc"));
			else
				learning_network.replace_features(my_experiment->get_float_param("replace_perc"));

			for (const auto &graph : graphs) {
				std::vector<std::string> cur_graphs;
				cur_graphs.push_back(std::to_string(my_experiment->get_int_param("run")));
				cur_graphs.push_back(std::to_string(step));
				cur_graphs.push_back(std::to_string(counter++));
				cur_graphs.push_back(std::to_string(graph.first.first));
				cur_graphs.push_back(std::to_string(graph.first.second));
				//cur_graphs.push_back(graph.second);
				cur_graphs.push_back("NA");
				correlation_metric.record_value(cur_graphs);
			}
		}

		if ( false && step % 1000 == 1 ) {
			std::vector<std::string> cur_weights;
			cur_weights.push_back(std::to_string(my_experiment->get_int_param("run")));
			cur_weights.push_back(std::to_string(step));
			auto current_weights = learning_network.get_prediction_weight_statistics();
			for (const auto &weight : current_weights)
				cur_weights.push_back(std::to_string(weight.first));
			for (const auto &weight : current_weights)
				cur_weights.push_back(std::to_string(weight.second));
			weight_metric.record_value(cur_weights);
		}

		auto input = input_sampler.get_random_vector(total_inputs);
		float pred = learning_network.forward(input);
		float target = target_network.forward(input);
		float error = target - pred;

		running_error = 0.995 * running_error + 0.005 * (target - pred) * (target - pred);
		learning_network.calculate_all_correlations();

		if (my_experiment->get_int_param("random_decorrelate")) {
			if ((my_experiment->get_int_param("age_restriction") && step > 25000) || !my_experiment->get_int_param("age_restriction")) {
				if (step % my_experiment->get_int_param("min_estimation_period") == 1) //update the random corr selections
					learning_network.update_random_correlation_selections(bool(my_experiment->get_int_param("age_restriction")),
					                                                      my_experiment->get_float_param("perc_of_total_pairs_to_estimate"));
				learning_network.calculate_random_correlations(my_experiment->get_int_param("min_estimation_period")); // update the random corr values
			}
		}

		// same random decorrelator as above but sampling based on correlations based on a single sample
		if (my_experiment->get_int_param("random_thresh_decorrelate")) {
			if ((my_experiment->get_int_param("age_restriction") && step > 25000) || !my_experiment->get_int_param("age_restriction")) {
				if (step % my_experiment->get_int_param("min_estimation_period") == 1) //update the random corr selections
					learning_network.update_random_correlation_selections_using_thresh(bool(my_experiment->get_int_param("age_restriction")));
				learning_network.calculate_random_correlations(my_experiment->get_int_param("min_estimation_period")); // update the random corr values
			}
		}

		learning_network.backward();
		//learning_network.update_parameters(error);
		//learning_network.update_parameters_only_prediction(error);
		learning_network.update_parameters_only_prediction(
        error,
        my_experiment->get_float_param("l2_lambda"),
        my_experiment->get_float_param("l1_lambda")
    );

		if (step%5000 == 1) {// || step%5000 == 4999){
			std::vector<std::string> cur_error;
			cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
			cur_error.push_back(std::to_string(step));
			cur_error.push_back(std::to_string(running_error));
			cur_error.push_back(std::to_string(learning_network.count_highly_correlated_features()));
			cur_error.push_back(std::to_string(learning_network.count_mature_features()));
			error_metric.record_value(cur_error);
			std::cout << "\nstep:" << step << std::endl;
			//print_vector(input);
			//print_vector(learning_network.get_prediction_weights());
			//print_vector(learning_network.get_feature_utilities());
			//print_vector(learning_network.get_prediction_gradients());
			std::cout << "target: " << target << " pred: " << pred << std::endl;
			std::cout << "running err: " << running_error << std::endl;
			//learning_network.print_all_correlations();
			//learning_network.print_all_statistics();
			std::cout << "count unremovable correlated features: " << learning_network.count_highly_correlated_features() << std::endl;
			std::cout << "count mature features" << learning_network.count_mature_features() << std::endl;
		}
		learning_network.zero_grad();
		if (step % 100000 == 1) {
			error_metric.commit_values();
			correlation_metric.commit_values();
			weight_metric.commit_values();
		}
	}
	std::vector<std::string> cur_error;
	cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
	cur_error.push_back(std::to_string(running_error));
	cur_error.push_back(std::to_string(learning_network.count_highly_correlated_features()));
	cur_error.push_back(std::to_string(learning_network.count_mature_features()));
	summary_metric.record_value(cur_error);
	summary_metric.commit_values();
	error_metric.commit_values();
	correlation_metric.commit_values();
	weight_metric.commit_values();
}
