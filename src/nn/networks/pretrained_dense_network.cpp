#include <cmath>
#include <execution>
#include <algorithm>
#include <iostream>
#include <utility>
#include "../../../include/nn/networks/pretrained_dense_network.h"

#include <torch/script.h>

using namespace torch::indexing;

PretrainedDenseNetwork::PretrainedDenseNetwork(int seed,
                                               int min_synapses_to_keep,
                                               int prune_interval,
                                               int start_pruning_at,
                                               float trace_decay_rate) {

	this->mt.seed(seed);
	this->min_synapses_to_keep = min_synapses_to_keep;
	this->prune_interval = prune_interval;
	this->start_pruning_at = start_pruning_at;
	this->trace_decay_rate = trace_decay_rate;
	this->total_initial_synapses = 0;
}


PretrainedDenseNetwork::~PretrainedDenseNetwork() = default;


void PretrainedDenseNetwork::load_linear_network(const torch::jit::script::Module& trained_model,
                                                 float step_size,
                                                 int no_of_input_features,
                                                 float utility_to_keep){
	for (int i = 0; i < no_of_input_features; i++) {
		SyncedNeuron *n = new LinearSyncedNeuron(true, false);
		n->neuron_age = 10000000;
		n->drinking_age = 0;
		n->set_layer_number(0);
		this->input_neurons.push_back(n);
		this->all_neurons.push_back(n);
	}

	int current_layer_number = 1;
	for (const auto& param_group : trained_model.parameters()) {
		std::vector<SyncedNeuron*> curr_layer;

		for (int neuron_idx = 0; neuron_idx < param_group.size(0); neuron_idx++) {
			SyncedNeuron *n;
			if (current_layer_number == trained_model.parameters().size()) {
				n = new LinearSyncedNeuron(false, true);
				this->output_neurons.push_back(n);
			}
			else
				n = new LinearSyncedNeuron(false, false);
			n->neuron_age = 0;
			n->drinking_age = 20000;
			n->set_layer_number(current_layer_number);
			this->all_neurons.push_back(static_cast<SyncedNeuron*>(n));
			curr_layer.push_back(static_cast<SyncedNeuron*>(n));

			for (int synapse_idx = 0; synapse_idx < param_group.size(1); synapse_idx++) {
				SyncedNeuron *source_neuron;
				if (current_layer_number > 1)
					source_neuron = this->all_neuron_layers[current_layer_number-2][synapse_idx];
				else
					source_neuron = this->input_neurons[synapse_idx];
				auto new_synapse = new SyncedSynapse(source_neuron,
				                                     n,
				                                     param_group.index({neuron_idx, synapse_idx}).item<float>(),
				                                     step_size);
				new_synapse->set_utility_to_keep(utility_to_keep);
				new_synapse->trace_decay_rate = trace_decay_rate;
				this->all_synapses.push_back(new_synapse);
			}
		}

		if (current_layer_number < trained_model.parameters().size())
			this->all_neuron_layers.push_back(curr_layer);
		if (current_layer_number > trained_model.parameters().size()) {
			std::cout << "shouldnt happen" <<std::endl;
			exit(1);
		}
		current_layer_number += 1;
	}
	this->total_initial_synapses = this->all_synapses.size();
}

void PretrainedDenseNetwork::load_relu_network(const torch::jit::script::Module& trained_model,
                                               float step_size,
                                               int no_of_input_features,
                                               float utility_to_keep) {
	for (int i = 0; i < no_of_input_features; i++) {
		SyncedNeuron *n = new LinearSyncedNeuron(true, false);
		n->neuron_age = 10000000;
		n->drinking_age = 0;
		n->set_layer_number(0);
		this->input_neurons.push_back(n);
		this->all_neurons.push_back(n);
	}

	int current_layer_number = 1;
	for (const auto& param_group : trained_model.parameters()) {
		std::vector<SyncedNeuron*> curr_layer;

		for (int neuron_idx = 0; neuron_idx < param_group.size(0); neuron_idx++) {
			SyncedNeuron *n;
			if (current_layer_number == trained_model.parameters().size()) {
				n = new SigmoidSyncedNeuron(false, true);
				this->output_neurons.push_back(n);
			}
			else
				n = new ReluSyncedNeuron(false, false);
			n->neuron_age = 0;
			n->drinking_age = 20000;
			n->set_layer_number(current_layer_number);
			this->all_neurons.push_back(static_cast<SyncedNeuron*>(n));
			curr_layer.push_back(static_cast<SyncedNeuron*>(n));

			for (int synapse_idx = 0; synapse_idx < param_group.size(1); synapse_idx++) {
				SyncedNeuron *source_neuron;
				if (current_layer_number > 1)
					source_neuron = this->all_neuron_layers[current_layer_number-2][synapse_idx];
				else
					source_neuron = this->input_neurons[synapse_idx];
				auto new_synapse = new SyncedSynapse(source_neuron,
				                                     n,
				                                     param_group.index({neuron_idx, synapse_idx}).item<float>(),
				                                     step_size);
				new_synapse->set_utility_to_keep(utility_to_keep);
				new_synapse->trace_decay_rate = trace_decay_rate;
				this->all_synapses.push_back(new_synapse);
			}
		}

		if (current_layer_number < trained_model.parameters().size())
			this->all_neuron_layers.push_back(curr_layer);
		if (current_layer_number > trained_model.parameters().size()) {
			std::cout << "shouldnt happen" <<std::endl;
			exit(1);
		}
		current_layer_number += 1;
	}
	this->total_initial_synapses = this->all_synapses.size();
}


void PretrainedDenseNetwork::print_synapse_status() {
	std::cout << "From\t\tTo\t\tWeight\t\tUtil_P\t\tDrop\t\tAct\t\tStep-size\t\tAge\n";
	for (auto it : this->all_synapses) {
		if (it->output_neuron->neuron_age > it->output_neuron->drinking_age
		    && it->input_neuron->neuron_age > it->input_neuron->drinking_age)
			std::cout << it->input_neuron->id << "\t\t" << it->output_neuron->id << "\t\t" << it->weight << "\t\t"
			          << it->synapse_utility << "\t\t" << it->dropout_utility_estimate << "\t\t" << it->activation_trace
			          << "\t\t" << it->step_size << "\t\t"
			          << it->age << std::endl;
	}
}


// Perform a forward pass layer by layer
void PretrainedDenseNetwork::forward(const std::vector<float>& inp) {
	this->set_input_values(inp);

	std::for_each(
		std::execution::unseq,
		this->input_neurons.begin(),
		this->input_neurons.end(),
		[&](SyncedNeuron *n) {
		n->fire(this->time_step);
	});

	int counter = 0;
	for (auto neuron_layer: this->all_neuron_layers) {
		counter++;
		std::for_each(
			std::execution::unseq,
			neuron_layer.begin(),
			neuron_layer.end(),
			[&](SyncedNeuron *n) {
			n->update_value(this->time_step);
		});
		std::for_each(
			std::execution::unseq,
			neuron_layer.begin(),
			neuron_layer.end(),
			[&](SyncedNeuron *n) {
			n->fire(this->time_step);
		});
	}

	std::for_each(
		std::execution::unseq,
		this->output_neurons.begin(),
		this->output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_value(this->time_step);
	});

	std::for_each(
		std::execution::unseq,
		this->output_neurons.begin(),
		this->output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->fire(this->time_step);
	});

	this->time_step++;
}


// Perform a backward pass and compute the gradients. This does not update the weights.
void PretrainedDenseNetwork::backward(std::vector<float> target) {
	this->introduce_targets(std::move(target));

	std::for_each(
		std::execution::unseq,
		output_neurons.begin(),
		output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->forward_gradients();
	});

	for (int layer = this->all_neuron_layers.size() - 1; layer >= 0; layer--) {
		std::for_each(
			std::execution::unseq,
			this->all_neuron_layers[layer].begin(),
			this->all_neuron_layers[layer].end(),
			[&](SyncedNeuron *n) {
			n->propagate_error();
		});
		std::for_each(
			std::execution::unseq,
			this->all_neuron_layers[layer].begin(),
			this->all_neuron_layers[layer].end(),
			[&](SyncedNeuron *n) {
			n->forward_gradients();
		});
	}

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->assign_credit();
	});
}


void PretrainedDenseNetwork::update_weights() {
	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_weight();
	});
}


void PretrainedDenseNetwork::update_utility_estimates(const std::string& pruner,
                                                      const std::vector<float>& input,
                                                      const std::vector<float>& prediction,
                                                      int dropout_iterations,
                                                      float dropout_perc){
	if (pruner == "utility_propagation")
		this->update_utility_propagation_estimates();
	else if (pruner == "activation_trace")
		this->update_activation_trace_estimates();
	else if (pruner == "dropout_utility_estimator")
		// we can have multiple fake forward passes to estimate the utilities using dropout
		for (int k = 0; k < dropout_iterations; k++)
			this->update_dropout_utility_estimates(input, prediction, dropout_perc);
}


void PretrainedDenseNetwork::update_activation_trace_estimates(){
	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_activation_trace();
	});
}


void PretrainedDenseNetwork::update_utility_propagation_estimates(){
	std::for_each(
		std::execution::unseq,
		this->all_neurons.begin(),
		this->all_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_utility();
	});

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_utility();
	});
}


void PretrainedDenseNetwork::update_dropout_utility_estimates(const std::vector<float>& inp,
                                                              std::vector<float> normal_predictions,
                                                              float dropout_perc){
	int total_dropped_synapses = this->all_synapses.size() * dropout_perc;
	if (total_dropped_synapses < 1)
		total_dropped_synapses = 1;

	// randomly sample <total_dropped_synapses> number of synapses to drop
	std::vector<SyncedSynapse *> synapses_to_drop;
	std::sample(this->all_synapses.begin(),
	            this->all_synapses.end(),
	            std::back_inserter(synapses_to_drop),
	            total_dropped_synapses,
	            this->mt);

	for (int i = 0; i < total_dropped_synapses; i++)
		synapses_to_drop[i]->is_dropped_out= true;
	// forward pass on the network with the dropped-out synapses
	//TODO this is bugged with non-zero step-sizes
	//TODO use the without sideeffects version of forward pass
	this->forward(inp);
	this->time_step--; //this forward pass is not an actual step
	auto dropout_predictions = this->read_output_values();

	// compute the absolute difference between the real prediction and the prediction from the dropped-out network
	float sum_of_differences = 0;
	for (int i = 0; i < dropout_predictions.size(); i++)
		sum_of_differences += std::fabs(normal_predictions[i] - dropout_predictions[i]);
	//sum_of_differences += fabs((dropout_predictions[i] - normal_predictions[i] )/ normal_predictions[i]);

	// update the estimates for the dropped-out synapses only
	for (int i = 0; i < total_dropped_synapses; i++) {
		synapses_to_drop[i]->dropout_utility_estimate = this->trace_decay_rate * synapses_to_drop[i]->dropout_utility_estimate + (1-this->trace_decay_rate) * sum_of_differences;
		synapses_to_drop[i]->is_dropped_out = false;
	}
}


// Select the weights to prune based on the dropout utility estimator algorithm
void PretrainedDenseNetwork::prune_using_dropout_utility_estimator() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return std::fabs(a->dropout_utility_estimate) < std::fabs(b->dropout_utility_estimate);
	} );

	for (int i = 0; i < total_removals; i++)
		all_synapses_copy[i]->is_useless = true;
}


// Select the weights to prune based on the trace of weight * activation
void PretrainedDenseNetwork::prune_using_trace_of_activation_magnitude() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return std::fabs(a->activation_trace) < std::fabs(b->activation_trace);
	} );

	for (int i = 0; i < total_removals; i++)
		all_synapses_copy[i]->is_useless = true;
}


// Select the weights to prune based on the absolute magnitude of the weight values
void PretrainedDenseNetwork::prune_using_weight_magnitude_pruner() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return std::fabs(a->weight) < std::fabs(b->weight);
	} );

	for (int i = 0; i < total_removals; i++) {
		all_synapses_copy[i]->is_useless = true;
	}
}


// Randomly sample weights and mark them for pruning
void PretrainedDenseNetwork::prune_using_random_pruner() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> synapses_to_remove;
	std::sample(this->all_synapses.begin(),
	            this->all_synapses.end(),
	            std::back_inserter(synapses_to_remove),
	            total_removals,
	            this->mt);

	for (int i = 0; i < total_removals; i++) {
		synapses_to_remove[i]->is_useless = true;
	}
}


// Use utility propagation algorithm to mark the least useless weights
void PretrainedDenseNetwork::prune_using_utility_propoagation() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return a->synapse_utility < b->synapse_utility;
	} );
	for (int i = 0; i < total_removals; i++) {
		all_synapses_copy[i]->is_useless = true;
	}
}


// Prune based on a schedule. This ensures that all pruners prune at the same rate
int PretrainedDenseNetwork::get_current_synapse_schedule() {
	return std::max(int(this->min_synapses_to_keep),
	                int( this->total_initial_synapses - ( (this->time_step - this->start_pruning_at) / this->prune_interval )));
}


void PretrainedDenseNetwork::prune_weights(std::string pruner){
	if (this->time_step > this->prune_interval &&
	    this->time_step > this->start_pruning_at &&
	    this->time_step % this->prune_interval == 0) {
		// If we pruned ahead of the schedule, we don't prune
		if (this->all_synapses.size() > this->get_current_synapse_schedule()) {
			if (pruner == "utility_propagation")
				this->prune_using_utility_propoagation();
			else if (pruner == "random")
				this->prune_using_random_pruner();
			else if (pruner == "weight_magnitude")
				this->prune_using_weight_magnitude_pruner();
			else if (pruner == "activation_trace")
				this->prune_using_trace_of_activation_magnitude();
			else if (pruner == "dropout_utility_estimator")
				this->prune_using_dropout_utility_estimator();
			else if (pruner != "none") {
				std::cout << "Invalid pruner specified" << std::endl;
				exit(1);
			}

			// mark weights to be removed based on the pruner
			std::for_each(
				this->all_neurons.begin(),
				this->all_neurons.end(),
				[&](SyncedNeuron *n) {
				n->mark_useless_weights();
			});

			// remove the marked weights
			std::for_each(
				this->all_neurons.begin(),
				this->all_neurons.end(),
				[&](SyncedNeuron *n) {
				n->prune_useless_weights();
			});

			//clean up the marked weights from all the other vectors as well
			for(auto & all_neuron_layer : this->all_neuron_layers) {

				auto it_n = std::remove_if(all_neuron_layer.begin(),
				                           all_neuron_layer.end(),
				                           to_delete_synced_n);
				if (it_n != all_neuron_layer.end()) {
					all_neuron_layer.erase(it_n, all_neuron_layer.end());
				}
			}

			auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_synced_s);
			this->all_synapses.erase(it, this->all_synapses.end());

			it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_synced_s);
			this->output_synapses.erase(it, this->output_synapses.end());

			auto it_n_2 = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_synced_n);
			this->all_neurons.erase(it_n_2, this->all_neurons.end());
		}
	}
}

