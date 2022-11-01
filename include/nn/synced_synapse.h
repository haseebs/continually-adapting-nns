#ifndef INCLUDE_NN_SYNCED_SYNAPSE_H_
#define INCLUDE_NN_SYNCED_SYNAPSE_H_


#include <vector>
#include <queue>
#include <utility>
#include "./dynamic_elem.h"
#include "./message.h"

class SyncedNeuron;

class SyncedSynapse : public dynamic_elem {
  bool is_recurrent_connection;

 public:
  static int64_t synapse_id_generator;
  int64_t id;

  bool is_useless;
  int age;
  float weight;
  float credit;
  float step_size;

  float utility_to_keep;
  float synapse_utility;
  float synapse_utility_to_distribute;
  float synapse_local_utility_trace = 0;
  bool propagate_gradients;
  float trace_decay_rate = 0.9999;
  bool disable_utility; //Also disables pruning
  float activation_trace;
  bool is_dropped_out;
  float dropout_utility_estimate;

  void update_activation_trace();

  message grad_queue;
  message grad_queue_weight_assignment;
  message_activation weight_assignment_past_activations;
  SyncedNeuron *input_neuron;
  SyncedNeuron *output_neuron;

  explicit SyncedSynapse(SyncedNeuron *input, SyncedNeuron *output, float w, float step_size);

  void block_gradients();

  void update_weight();

  void update_utility();

  void memory_leak_patch();

  void assign_credit();

  void set_utility_to_keep(float);

  float get_utility_to_keep();

  ~SyncedSynapse() = default;
};

#endif //INCLUDE_NN_SYNCED_SYNAPSE_H_
