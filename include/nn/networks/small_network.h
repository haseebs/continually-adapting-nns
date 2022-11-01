#ifndef INCLUDE_NN_NETWORKS_PRETRAINED_DENSE_H_
#define INCLUDE_NN_NETWORKS_PRETRAINED_DENSE_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synced_synapse.h"
#include "../synced_neuron.h"
#include "../dynamic_elem.h"
#include "./synced_network.h"
#include <torch/script.h>

class SmallNetwork: public SyncedNetwork {

 public:

  std::vector<SyncedSynapse *> active_synapses;

  std::vector<std::vector<SyncedNeuron *>> all_neuron_layers;

  SmallNetwork(float step_size,
               int seed,
               int no_of_input_features);

  ~SmallNetwork();

  void print_graph(SyncedNeuron *root);

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets);

  void update_weights();
};


#endif //INCLUDE_NN_NETWORKS_DENSE_NETWORK_H_
