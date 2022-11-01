#include "../../include/nn/synced_neuron.h"
#include <random>
#include <vector>

SyncedNeuron::SyncedNeuron(bool is_input, bool is_output) {
	value = 0;
	value_before_firing = 0;
	id = neuron_id_generator;
	useless_neuron = false;
	neuron_id_generator++;
	this->is_output_neuron = is_output;
	is_input_neuron = is_input;
	neuron_age = 0;
	references = 0;
	neuron_utility = 0;
	drinking_age = 5000;
	mark_useless_prob = 0.99;
	is_bias_unit = false;
	is_mature = false;
	sum_of_utility_traces = 0;
}


void SyncedNeuron::set_layer_number(int layer) {
	this->layer_number = layer;
}


int SyncedNeuron::get_layer_number() const {
	return this->layer_number;
}


// Update utility for utility propagation algorithm
void SyncedNeuron::update_utility() {

	this->neuron_utility = 0;
	for (auto it: this->outgoing_synapses) {
		this->neuron_utility += it->synapse_utility_to_distribute;
	}
	if (this->is_output_neuron)
		this->neuron_utility = 1;

	this->sum_of_utility_traces = 0;
	for (auto it: this->incoming_synapses) {
		if (!it->disable_utility)
			this->sum_of_utility_traces += it->synapse_local_utility_trace;
	}
}


// Forward pass through a neuron (apply activation to linear combinations)
void SyncedNeuron::fire(int time_step) {
	this->neuron_age++;
	if (this->neuron_age > drinking_age)
		this->is_mature = true;
	this->value = this->forward(value_before_firing);
}


// Compute the linear combination of all the incoming connections to the neuron
void SyncedNeuron::update_value(int time_step) {

	this->value_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
	for (auto &it : this->incoming_synapses) {
		it->age++;
		if (!it->is_dropped_out)
			this->value_before_firing += it->weight * it->input_neuron->value;
	}
}


bool to_delete_ss(SyncedSynapse *s) {
	return s->is_useless;
}


/**
 * For each incoming synapse of a neuron, add the gradient from the error in this
 * neuron to its grad_queue for weight assignment. If we do pass gradients backwards,
 * also pass the gradient from the error to grad_queue for use in back propagation.
 */
void SyncedNeuron::forward_gradients() {
//  If this neuron has gradients to pass back
	for (auto &it : this->incoming_synapses) {
//          We pack our gradient into a new message and pass it back to our incoming synapse.
		message grad_temp(this->error_gradient.gradient, this->error_gradient.time_step);
		if (it->propagate_gradients)
			it->grad_queue = grad_temp;
		it->grad_queue_weight_assignment = grad_temp;
	} //  Remove this gradient from our list of things needed to pass back
}



int SyncedNeuron::get_no_of_syanpses_with_gradients() {
	int synapse_with_gradient = 0;
	for (auto it: this->outgoing_synapses) {
		if (it->propagate_gradients)
			synapse_with_gradient++;
	}
	return synapse_with_gradient;
}


void SyncedNeuron::propagate_error() {
	float accumulate_gradient = 0;
	int grad_queue_timestep;
//  No gradient propagation required for prediction nodes

// We need a loop invariant for this function to make sure progress is always made. Things to make sure:
// 1. A queue for a certain outgoing path won't grow large indefinitely
// 2. Adding new connections or removing old connections won't cause deadlocks
// 3. We can never get in a situation in which neither activation nor the gradient is popped. Some number should strictly increase or decrease

// No need to pass gradients if there are no out-going nodes with gradients
	if (this->get_no_of_syanpses_with_gradients() > 0 && !is_input_neuron) {

		for (auto &output_synapses_iterator : this->outgoing_synapses) {
			accumulate_gradient += output_synapses_iterator->weight *
			                       output_synapses_iterator->grad_queue.gradient *
			                       this->backward(this->value);
			grad_queue_timestep = output_synapses_iterator->grad_queue.time_step;
			output_synapses_iterator->grad_queue.remove = true;

		}

		message n_message(accumulate_gradient, grad_queue_timestep);

		this->error_gradient = n_message;
	}
}

void SyncedNeuron::mark_useless_weights() {

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it and its incoming synapses.
	if (this->incoming_synapses.empty() && !this->is_input_neuron && !this->is_output_neuron) {
		this->useless_neuron = true;
		for (auto it : this->outgoing_synapses)
			it->is_useless = true;
	}

	if (this->outgoing_synapses.empty() && !this->is_output_neuron && !this->is_input_neuron) {
		this->useless_neuron = true;
		for (auto it : this->incoming_synapses)
			it->is_useless = true;
	}
}


/**
 * Delete outgoing and incoming synapses that were marked earlier as is_useless.
 */
void SyncedNeuron::prune_useless_weights() {
	std::for_each(
//            std::execution::seq,
		this->outgoing_synapses.begin(),
		this->outgoing_synapses.end(),
		[&](SyncedSynapse *s) {
		if (s->is_useless) {
			s->decrement_reference();
			if (s->input_neuron != nullptr) {
				s->input_neuron->decrement_reference();
				s->input_neuron = nullptr;
			}
			if (s->output_neuron != nullptr) {
				s->output_neuron->decrement_reference();
				s->output_neuron = nullptr;
			}
		}
	});

	auto it = std::remove_if(this->outgoing_synapses.begin(), this->outgoing_synapses.end(), to_delete_ss);
	this->outgoing_synapses.erase(it, this->outgoing_synapses.end());

	std::for_each(
//            std::execution::seq,
		this->incoming_synapses.begin(),
		this->incoming_synapses.end(),
		[&](SyncedSynapse *s) {
		if (s->is_useless) {
			s->decrement_reference();
			if (s->input_neuron != nullptr) {
				s->input_neuron->decrement_reference();
				s->input_neuron = nullptr;
			}
			if (s->output_neuron != nullptr) {
				s->output_neuron->decrement_reference();
				s->output_neuron = nullptr;
			}
		}
	});
	it = std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete_ss);
	this->incoming_synapses.erase(it, this->incoming_synapses.end());
}


/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @return: squared error
 */
float SyncedNeuron::introduce_targets(float target, int time_step) {
	float error = this->value - target;

	message m(this->backward(this->value) * error, time_step);
	this->error_gradient = m;
	return error * error;
}


float LinearSyncedNeuron::forward(float temp_value) {
	return temp_value;
}


float LinearSyncedNeuron::backward(float post_activation) {
	return 1;
}


float ReluSyncedNeuron::forward(float temp_value) {

	if (temp_value <= 0)
		return 0;

	return temp_value;
}


float ReluSyncedNeuron::backward(float post_activation) {
	if (post_activation > 0)
		return 1;
	else
		return 0;
}


float CenteredSigmoidSyncedNeuron::forward(float temp_value) {
	//Note: this is different from normal sigmoid
	return sigmoid(temp_value) - 0.5;
}


float CenteredSigmoidSyncedNeuron::backward(float post_activation) {
	return post_activation * (1 - post_activation);
}


float SigmoidSyncedNeuron::forward(float temp_value) {
	return sigmoid(temp_value);
}


float SigmoidSyncedNeuron::backward(float post_activation) {
	return post_activation * (1 - post_activation);
}


float BiasSyncedNeuron::forward(float temp_value) {
	return 1;
}


float BiasSyncedNeuron::backward(float output_grad) {
	return 0;
}


float LTUSynced::forward(float temp_value) {
	if (temp_value >= this->activation_threshold)
		return 1;
	return 0;
}


float LTUSynced::backward(float output_grad) {
	return 0;
}


void LTUSynced::set_threshold(float threshold) {
	this->activation_threshold = threshold;
}


ReluSyncedNeuron::ReluSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {
}


CenteredSigmoidSyncedNeuron::CenteredSigmoidSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {
}


SigmoidSyncedNeuron::SigmoidSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {
}


LTUSynced::LTUSynced(bool is_input, bool is_output, float threshold) : SyncedNeuron(is_input, is_output) {
	this->activation_threshold = threshold;
}


BiasSyncedNeuron::BiasSyncedNeuron() : SyncedNeuron(false, false) {
	this->is_bias_unit = true;
}


LinearSyncedNeuron::LinearSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {
}


std::mt19937 SyncedNeuron::gen = std::mt19937(0);


int64_t SyncedNeuron::neuron_id_generator = 0;
