#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>

#include "../include/utils.h"
#include "../include/nn/networks/small_network.h"
#include "../include/experiment/Experiment.h"
#include "../include/experiment/Metric.h"


int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector < std::string > {"run", "step", "error"},
                               std::vector < std::string > {"int", "int", "real"},
                               std::vector < std::string > {"run", "step"});

  Metric feature_metric = Metric(my_experiment->database_name, "feature_table",
                               std::vector < std::string > {"run", "step", "w1", "w2", "w3"},
                               std::vector < std::string > {"int", "int", "real", "real", "real"},
                               std::vector < std::string > {"run", "step"});

  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment->get_int_param("seed"));
  std::uniform_real_distribution<float> input_sampler(-1,1);
  std::vector<float> inputs;

	SmallNetwork network = SmallNetwork(my_experiment->get_float_param("step_size"),
                                      my_experiment->get_int_param("seed"),
                                      3);


  network.print_neuron_status();
  network.print_synapse_status();

  inputs.push_back(0);
  inputs.push_back(input_sampler(mt));
  inputs.push_back(input_sampler(mt));

  std::vector<float> target;
  target.push_back(inputs[1] + 3*inputs[2]);
  target.push_back(inputs[1] + 3*inputs[2]);
  float running_error = 0.05;
  float running_error2 = 0.05;

  for (int step = 0; step < my_experiment->get_int_param("steps"); step++) {

    inputs[0] = 0;
    inputs[1] = input_sampler(mt);
    inputs[2] = input_sampler(mt);
    target[0] = sigmoid(inputs[1] + 3*inputs[2]);
    target[1] = sigmoid(3* inputs[1] + inputs[2]);

    network.forward(inputs);
    auto prediction = network.read_output_values();
    network.backward(target);

    float error = (prediction[0] - target[0]) * (prediction[0] - target[0]);
    float error2 = (prediction[1] - target[1]) * (prediction[1] - target[1]);
    running_error = 0.99 * running_error + 0.01 * error;
    running_error2 = 0.99 * running_error2 + 0.01 * error2;

    if (step%1000 == 0){
      std::cout << "\nstep:" << step << std::endl;
      std::cout << "Error: " << running_error << std::endl;
      std::cout << "Error(2): " << running_error2 << std::endl;
      print_vector(inputs);
      network.print_synapse_status();
      std::cout << "target: ";
      print_vector(target);
      std::cout  << " pred: ";
      print_vector(prediction);
      std::cout << std::endl;
    }
    network.update_weights();
  }
}
