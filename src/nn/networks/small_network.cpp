#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/small_network.h"

#include <torch/script.h>


SmallNetwork::SmallNetwork(float step_size,
                           int seed,
                           int no_of_input_features) {

	this->mt.seed(seed);
	std::uniform_real_distribution<float> weight_sampler(0, 1);

	int HIDDEN_NEURONS = 2;

	for (int i = 0; i < no_of_input_features; i++) {
		SyncedNeuron *n = new LinearSyncedNeuron(true, false);
		n->set_layer_number(0);
		this->input_neurons.push_back(n);
		this->all_neurons.push_back(n);
	}

	std::vector<SyncedNeuron*> curr_layer;
	for (int b = 0; b < HIDDEN_NEURONS; b++) {
		SyncedNeuron *n = new SigmoidSyncedNeuron(false, false);
		n->set_layer_number(1);
		this->all_neurons.push_back(n);
		curr_layer.push_back(n);
	}
	this->all_neuron_layers.push_back(curr_layer);

	for (int outer = 0; outer < no_of_input_features; outer++) {
		for (int inner = 0; inner < HIDDEN_NEURONS; inner++) {
			//SyncedSynapse *s = new SyncedSynapse(this->input_neurons[outer], this->all_neuron_layers[0][inner], weight_sampler(this->mt), step_size);
			SyncedSynapse *s = new SyncedSynapse(this->input_neurons[outer], this->all_neuron_layers[0][inner], weight_sampler(this->mt), step_size);
			this->all_synapses.push_back(s);
		}
	}

	for (int outer = 0; outer < 2; outer++) {
		SyncedNeuron *output_neuron = new LinearSyncedNeuron(false, true);
		output_neuron->set_layer_number(100);
		this->output_neurons.push_back(output_neuron);
		this->all_neurons.push_back(output_neuron);

		for (int inner = 0; inner < HIDDEN_NEURONS; inner++) {
			//if (outer != inner)
			//  continue;
			SyncedSynapse *s = new SyncedSynapse(this->all_neuron_layers[0][inner], output_neuron, weight_sampler(this->mt), step_size);
			//this->output_synapses.push_back(s);
			this->all_synapses.push_back(s);
		}
	}

	//for (int outer = 0; outer < no_of_input_features; outer++) {
	//  for (int inner = 0; inner < 2; inner++){
	//    SyncedSynapse *s = new SyncedSynapse(this->input_neurons[outer], this->output_neurons[inner], 0, step_size);
	//    this->all_synapses.push_back(s);
	//  }
	//}
	//
	//

}

SmallNetwork::~SmallNetwork() {
}

void SmallNetwork::forward(std::vector<float> inp) {

	this->set_input_values(inp);

	std::for_each(
		std::execution::unseq,
		this->input_neurons.begin(),
		this->input_neurons.end(),
		[&](SyncedNeuron *n) {
		n->fire(this->time_step);
	});

	int counter = 0;
	for (auto LTU_neuron_list: this->all_neuron_layers) {
		counter++;
		std::for_each(
			std::execution::unseq,
			LTU_neuron_list.begin(),
			LTU_neuron_list.end(),
			[&](SyncedNeuron *n) {
			n->update_value(this->time_step);
		});

		std::for_each(
			std::execution::unseq,
			LTU_neuron_list.begin(),
			LTU_neuron_list.end(),
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

	std::for_each(
		std::execution::unseq,
		this->all_neurons.begin(),
		this->all_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_utility();
	});

	this->time_step++;
}


void SmallNetwork::backward(std::vector<float> target) {
	this->introduce_targets(target);

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
//  Calculate our credit

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_activation_trace();
	});

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_utility();
	});

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->assign_credit();
	});
}

void SmallNetwork::update_weights() {
	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_weight();
	});
}
