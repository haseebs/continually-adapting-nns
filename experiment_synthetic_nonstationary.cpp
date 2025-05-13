#include "include/experiment/Experiment.h"
#include "include/logger/logger.h"
#include "include/nn/networks/single_layer_network.h"
#include "include/nn/synced_neuron.h"
#include <cmath>
#include <random>
#include <string>
#include "include/stdout.h"

int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);
  if (my_experiment->get_int_param("sum_features") &&
      !my_experiment->get_int_param("n2_decorrelate"))
    exit(1);

  Logger *logger;
  logger = new MongoDBLogger("mongodb://hshah1:a1b2c3d4@127.0.0.1:27017/",
                             my_experiment->get_string_param("collection"),
                             my_experiment->get_string_param("database"));

  nlohmann::json j;
  for (auto const &x : my_experiment->args_for_run) {
    j[x.first] = x.second;
  }

  bool to_print = true;

  std::mt19937 mt(my_experiment->get_int_param("seed"));
  int total_inputs = my_experiment->get_int_param("n_inputs") +
                     my_experiment->get_int_param("n_distractors");
  auto target_network = SingleLayerNetwork(
      0.0, my_experiment->get_int_param("seed") + 1000,
      my_experiment->get_int_param("n_inputs"),
      my_experiment->get_int_param("n_target_features"), true);
  SyncedNeuron::neuron_id_generator = 0;
  auto learning_network = SingleLayerNetwork(
      my_experiment->get_float_param("step_size"),
      my_experiment->get_int_param("seed"), total_inputs,
      my_experiment->get_int_param("n_learner_features"), false);

  auto input_sampler = uniform_random(my_experiment->get_int_param("seed"), -10, 10);
  std::vector<std::pair<int, float>> error_data;
  std::vector<std::pair<int, float>> n_correlated_data;
  std::vector<std::pair<int, float>> n_mature_data;
  std::vector<std::pair<int, std::pair<int, float>>> real_correlation;
  std::vector<std::pair<int, std::pair<int, float>>> estimated_correlation;
  float running_error = 0.05;
  std::vector<std::pair<std::pair<float, float>, std::string>> graphs;
  int counter = 0;

  for (int step = 0; step < my_experiment->get_int_param("steps"); step++) {
    // replace features
    if (step % my_experiment->get_int_param("replace_every") == 1) {
      if (my_experiment->get_int_param("n2_decorrelate"))
        graphs = learning_network.replace_features_n2_decorrelator_v3(
                    my_experiment->get_float_param("replace_perc"),
                    bool(my_experiment->get_int_param("sum_features")),
                    my_experiment->get_float_param("decorrelate_perc"),
                    bool(my_experiment->get_int_param("use_generate_and_test")));

      else if (my_experiment->get_int_param("random_decorrelate") || my_experiment->get_int_param("random_thresh_decorrelate"))
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

		if (step % my_experiment->get_int_param("change_target_steps") == 0) {
			target_network = SingleLayerNetwork(
					0.0, my_experiment->get_int_param("seed") + 1000 + step,
					my_experiment->get_int_param("n_inputs"),
					my_experiment->get_int_param("n_target_features"), true);
		}

    auto input = input_sampler.get_random_vector(total_inputs);
    float pred = learning_network.forward(input);
    float target = target_network.forward(input);
    float error = target - pred;

    learning_network.calculate_all_correlations();
    running_error =
        0.995 * running_error + 0.005 * (target - pred) * (target - pred);

    if (my_experiment->get_int_param("random_decorrelate")) {
      if ((my_experiment->get_int_param("age_restriction") && step > 25000) || !my_experiment->get_int_param("age_restriction")) {
        // update the random corr selections
        if (step % my_experiment->get_int_param("min_estimation_period") == 1)
          learning_network.update_random_correlation_selections(
              bool(my_experiment->get_int_param("age_restriction")),
              my_experiment->get_float_param("perc_of_total_pairs_to_estimate"));
        // update the random corr values
        learning_network.calculate_random_correlations(
            my_experiment->get_int_param("min_estimation_period"));
      }
    }

    learning_network.backward();
    if (my_experiment->get_int_param("freeze_weights"))
      learning_network.update_parameters_only_prediction_RMSProp(error);
    else
      learning_network.update_parameters(error);

    if (step % 1000 == 1)
    {
      if(to_print) ansii::printKeyValue(10, 9, "Current error", std::to_string(error));
      if (to_print) ansii::printKeyValue(10, 5, "Average Error", std::to_string(running_error));
      if (to_print) ansii::printKeyValue(10, 6, "Step", std::to_string(step));
      if(to_print) ansii::printKeyValue(10, 7, "Correlated Features", std::to_string(learning_network.count_highly_correlated_features()));
      if(to_print) ansii::printKeyValue(10, 8, "Mature Features", std::to_string(learning_network.count_mature_features()));
    }

    if (step % 5000 == 1) {
      error_data.push_back(std::make_pair(step, running_error));
      n_correlated_data.push_back(std::make_pair(step, learning_network.count_highly_correlated_features()));
      n_mature_data.push_back(std::make_pair(step, learning_network.count_mature_features()));
    }
    learning_network.zero_grad();
  }
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
