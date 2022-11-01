#include "../../include/nn/synced_synapse.h"
#include <math.h>
#include <vector>
#include <iostream>
#include "../../include/nn/synced_neuron.h"

int64_t SyncedSynapse::synapse_id_generator = 0;

SyncedSynapse::SyncedSynapse(SyncedNeuron *input, SyncedNeuron *output, float w, float step_size) {
	references = 0;
	input_neuron = input;
	input->increment_reference();
	output_neuron = output;
	output->increment_reference();
	credit = 0;
	is_useless = false;
	age = 0;
	weight = w;
	this->step_size = step_size;
	this->increment_reference();
	input_neuron->outgoing_synapses.push_back(this);
	this->increment_reference();
	output_neuron->incoming_synapses.push_back(this);
	this->id = synapse_id_generator;
	synapse_id_generator++;
	propagate_gradients = true;
	synapse_utility = 0;
	if (input->is_input_neuron) {
		propagate_gradients = false;
	}
	utility_to_keep = 0.000001;
	disable_utility = false;
	this->trace_decay_rate = 0.999;
	this->synapse_local_utility_trace = 0;
	this->synapse_utility_to_distribute = 0;
	this->activation_trace = 0;
	this->is_dropped_out = false;
	//TODO compare initializations
	this->dropout_utility_estimate = 0;
}
//

void SyncedSynapse::update_activation_trace() {
	this->activation_trace = this->trace_decay_rate * this->activation_trace + (1-trace_decay_rate) * fabs(this->input_neuron->value * this->weight);
}

void SyncedSynapse::set_utility_to_keep(float util) {
	this->utility_to_keep = util;
}

float SyncedSynapse::get_utility_to_keep() {
	return this->utility_to_keep;
}

void SyncedSynapse::update_utility() {

	float diff = this->output_neuron->value - this->output_neuron->forward(
		this->output_neuron->value_before_firing - this->input_neuron->value * this->weight);
//  0.999 is a hyper-parameter.
	if (!this->disable_utility) {
		this->synapse_local_utility_trace = this->trace_decay_rate * this->synapse_local_utility_trace + (1-this->trace_decay_rate) * std::abs(diff);
		this->synapse_utility =
			(synapse_local_utility_trace * this->output_neuron->neuron_utility)
			/ (this->output_neuron->sum_of_utility_traces + 1e-10);
		if (this->synapse_utility > this->utility_to_keep) {
			this->synapse_utility_to_distribute = this->synapse_utility - this->utility_to_keep;
			//TODO uncomment to keep utility
			//this->synapse_utility = this->utility_to_keep;
		} else {
			this->synapse_utility_to_distribute = 0;
		}
	} else {
		this->synapse_utility = 0;
		this->synapse_utility_to_distribute = 0;
		this->synapse_local_utility_trace = 0;
	}
}

/**
 * Calculate and set credit based on gradients in the current synapse.
 */
void SyncedSynapse::assign_credit() {
	this->credit = this->input_neuron->value * this->grad_queue_weight_assignment.gradient;
}

void SyncedSynapse::block_gradients() {
	propagate_gradients = false;
}

void SyncedSynapse::update_weight() {
	this->weight -= (this->step_size * this->credit);
}

