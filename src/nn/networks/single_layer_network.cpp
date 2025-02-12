#include <execution>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>
#include <utility>
#include <string>
#include "../../../include/nn/networks/single_layer_network.h"
#include "../../../include/nn/synced_neuron.h"

//
SingleLayerNetwork::SingleLayerNetwork(float step_size,
                                       int seed,
                                       int no_of_input_features,
                                       int no_of_intermediate_features,
                                       bool is_target_network)
{
	this->step_size = step_size;
	this->v_old = 0;
	this->std_cap = 0.001;
	this->mt.seed(seed);
	this->second_mt.seed(seed + 9999);
	this->third_mt.seed(seed + 99999);
	std::uniform_real_distribution<float> weight_sampler(-1, 1);
	std::uniform_real_distribution<float> prob_sampler(0, 1);
	std::uniform_int_distribution<int> index_sampler(0, no_of_input_features - 1);

	for (int i = 0; i < no_of_input_features; i++)
	{
		LinearSyncedNeuron* n = new LinearSyncedNeuron(true, false);
		this->input_neurons.push_back(n);
	}

	//  for (int i = 0; i < 3; i++) {
	//    SigmoidSyncedNeuron *new_feature = new SigmoidSyncedNeuron(false, false);
	//    this->intermediate_neurons.push_back(new_feature);
	//      SyncedSynapse *new_synapse = new SyncedSynapse(this->input_neurons[i],
	//                                         new_feature,
	//                                         1,
	//                                         step_size);
	//      this->input_synapses.push_back(new_synapse);
	//  }

	for (int i = 0; i < no_of_intermediate_features; i++)
	{
		SigmoidSyncedNeuron* new_feature = new SigmoidSyncedNeuron(false, false);
		this->intermediate_neurons.push_back(new_feature);

		std::set<int> connection_indices;
		int max_connections = index_sampler(mt) + 1;
		while (connection_indices.size() < max_connections)
		{
			int new_index = index_sampler(mt);
			if (connection_indices.contains(new_index))
				continue;
			connection_indices.insert(new_index);
			float new_weight = weight_sampler(mt);
			SyncedSynapse* new_synapse = new SyncedSynapse(this->input_neurons[new_index],
			                                               new_feature,
			                                               new_weight,
			                                               step_size);
			this->input_synapses.push_back(new_synapse);
		}
	}

	for (int counter = 0; counter < this->intermediate_neurons.size(); counter++)
	{
		for (int inner_counter = 0; inner_counter < this->intermediate_neurons[counter]->incoming_synapses.size();
		     inner_counter++)
		{
			// std::cout << this->intermediate_neurons[counter]->incoming_synapses[inner_counter]->input_neuron->id << "\tto\t"
			// << this->intermediate_neurons[counter]->id << std::endl;
		}
	}

	predictions = 0;
	bias = 0;
	bias_gradients = 0;
	for (int j = 0; j < this->intermediate_neurons.size(); j++)
	{
		if (is_target_network)
			prediction_weights.push_back(weight_sampler(mt));
		else
			prediction_weights.push_back(0);
		prediction_weights_gradient.push_back(0);
		prediction_weights_gradient_trace.push_back(0);
		feature_utility_trace.push_back(0); // trace of normalized feature value * outgoing weight
		feature_mean.push_back(0);
		feature_std.push_back(1);
		id_to_idx[intermediate_neurons[j]->id] = j; //TODO this isnt cleaned up
	}
}

std::string SingleLayerNetwork::get_graph(int id1, int id2)
{
	return "NA";
	std::string dot_string = "digraph network{\n"
		"\tnode [shape = circle];\n";

	for (auto& it : input_neurons)
	{
		for (auto& os : it->outgoing_synapses)
		{
			//if (os->input_neuron->is_mature && os->output_neuron->is_mature) {
			auto current_n = os;
			if (current_n->output_neuron->id == id1 || current_n->output_neuron->id == id2)
			{
				dot_string += "\t" + std::to_string(current_n->input_neuron->id)
					+ "->" + std::to_string(current_n->output_neuron->id) //+ ";\n";
					//+ ";\n";
					+ "[label = \"" + std::to_string(os->weight).substr(0, std::to_string(os->weight).find(".") + 2 + 1)
					+ "\"];\n";
			}
		}
	}

	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		if (intermediate_neurons[i]->id == id1 || intermediate_neurons[i]->id == id2)
			dot_string += "\t" + std::to_string(intermediate_neurons[i]->id)
				+ "->" + "y" //+ ";\n";
				+ "[label = \"" + std::to_string(prediction_weights[i]).substr(
					0, std::to_string(prediction_weights[i]).find(".") + 2 + 1) + "\"];\n";
	}

	dot_string += "\n}";
	return dot_string;
}


void SingleLayerNetwork::replace_features_with_idx(int feature_idx)
{
	std::uniform_real_distribution<float> weight_sampler(-1, 1);
	std::uniform_real_distribution<float> prob_sampler(0, 1);
	std::uniform_int_distribution<int> index_sampler(0, this->input_neurons.size() - 1);
	float median_feature_utility = median(feature_utility_trace);
	int old_feature_id = this->intermediate_neurons[feature_idx]->id;

	for (auto& synapse : this->intermediate_neurons[feature_idx]->incoming_synapses)
		synapse->is_useless = true;

	SigmoidSyncedNeuron* new_feature = new SigmoidSyncedNeuron(false, false);
	this->intermediate_neurons[feature_idx] = new_feature;

	std::set<int> connection_indices;
	int max_connections = index_sampler(mt) + 1;
	while (connection_indices.size() < max_connections)
	{
		int new_index = index_sampler(mt);
		if (connection_indices.contains(new_index))
			continue;
		connection_indices.insert(new_index);
		float new_weight = weight_sampler(mt);
		SyncedSynapse* new_synapse = new SyncedSynapse(this->input_neurons[new_index],
		                                               new_feature,
		                                               new_weight,
		                                               step_size);
		this->input_synapses.push_back(new_synapse);
	}
	prediction_weights[feature_idx] = 0;
	prediction_weights_gradient[feature_idx] = 0;
	prediction_weights_gradient_trace[feature_idx] = 0;
	feature_utility_trace[feature_idx] = median_feature_utility;
	feature_mean[feature_idx] = 0;
	feature_std[feature_idx] = 1;
	id_to_idx[intermediate_neurons[feature_idx]->id] = feature_idx;

	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		auto id_pair = std::make_pair(intermediate_neurons[i]->id, old_feature_id);
		feature_correlations.erase(id_pair);
		random_feature_correlations.erase(id_pair);
		random_feature_correlations_ages.erase(id_pair);
		id_pair = std::make_pair(old_feature_id, intermediate_neurons[i]->id);
		feature_correlations.erase(id_pair);
		random_feature_correlations.erase(id_pair);
		random_feature_correlations_ages.erase(id_pair);
	}
	id_to_idx.erase(old_feature_id);

	auto it = std::remove_if(this->input_synapses.begin(), this->input_synapses.end(), [](SyncedSynapse* s)
	{
		return s->is_useless;
	});
	this->input_synapses.erase(it, this->input_synapses.end());
	for (auto& n : this->input_neurons)
	{
		auto it = std::remove_if(n->outgoing_synapses.begin(), n->outgoing_synapses.end(), [](SyncedSynapse* s)
		{
			return s->is_useless;
		});
		n->outgoing_synapses.erase(it, n->outgoing_synapses.end());
	}
	// std::for_each(
	// 	std::execution::seq,
	// 	this->input_neurons.begin(),
	// 	this->input_neurons.end(),
	// 	[&](SyncedNeuron *n) {
	// 	auto it = std::remove_if(n->outgoing_synapses.begin(), n->outgoing_synapses.end(), [](SyncedSynapse *s){
	// 		return s->is_useless;
	// 	});
	// 	n->outgoing_synapses.erase(it, n->outgoing_synapses.end());
	// });
}


float SingleLayerNetwork::get_normalized_value(int idx)
{
	return (intermediate_neurons[idx]->value - feature_mean[idx]) / sqrt(feature_std[idx]);
}


void SingleLayerNetwork::calculate_all_correlations()
{
	//TODO the correlations map needs to be cleaned up for removed features
	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		for (int j = i + 1; j < intermediate_neurons.size(); j++)
		{
			auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
			if (feature_correlations.contains(id_pair))
				feature_correlations[id_pair] = 0.9999 * feature_correlations[id_pair] + 0.0001 * get_normalized_value(i)
					* get_normalized_value(j);
			else
				//feature_correlations[id_pair] = std::max((float)-1.0, std::min((float)1.0, get_normalized_value(i)*get_normalized_value(j)));
				feature_correlations[id_pair] = 0;
		}
	}
}


void SingleLayerNetwork::update_random_correlation_selections_using_thresh(bool age_restriction)
{
	std::vector<intpair> expired_pairs; //pairs that we gotta stop considering
	for (auto& [id_pair, corr] : random_feature_correlations)
	{
		if (fabs(corr) < 0.85)
			expired_pairs.push_back(id_pair);
	}

	for (auto& id_pair : expired_pairs)
	{
		random_feature_correlations.erase(id_pair);
		random_feature_correlations_ages.erase(id_pair);
	}

	int total_sampled = 0;
	int max_samples = intermediate_neurons.size() * intermediate_neurons.size() / 2;
	std::uniform_int_distribution<int> index_sampler(0, this->intermediate_neurons.size() - 1);
	while (random_feature_correlations.size() < intermediate_neurons.size())
	{
		int i = index_sampler(mt);
		int j = index_sampler(mt);

		if (total_sampled >= max_samples)
			break;
		//smarter sampler
		if (fabs(get_normalized_value(i) * get_normalized_value(j)) < 0.85)
		{
			total_sampled++;
			continue;
		}
		// ensure ordering of idxes in pairs so we dont have to check both
		if (i == j)
			continue;
		if (i > j)
			std::swap(i, j);

		auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
		if (random_feature_correlations.contains(id_pair) || (age_restriction && (intermediate_neurons[i]->neuron_age <
			4500 || intermediate_neurons[j]->neuron_age < 4500)))
		{
			total_sampled++;
			continue;
		}
		random_feature_correlations[id_pair] = 0;
		//random_feature_correlations[id_pair] = std::max((float)-1.0, std::min((float)1.0, get_normalized_value(i)*get_normalized_value(j)));
		random_feature_correlations_ages[id_pair] = 0;
	}
	// std::cout << "sampled: " << total_sampled << std::endl;
}


void SingleLayerNetwork::update_random_correlation_selections(bool age_restriction,
                                                              float perc_of_total_pairs_to_estimate)
{
	std::vector<intpair> expired_pairs; //pairs that we gotta stop considering
	for (auto& [id_pair, corr] : random_feature_correlations)
	{
		if (fabs(corr) < 0.85)
			expired_pairs.push_back(id_pair);
	}

	for (auto& id_pair : expired_pairs)
	{
		random_feature_correlations.erase(id_pair);
		random_feature_correlations_ages.erase(id_pair);
	}

	std::uniform_int_distribution<int> index_sampler(0, this->intermediate_neurons.size() - 1);
	int total_pair_selections = intermediate_neurons.size(); // just consider n pairs
	if (perc_of_total_pairs_to_estimate != 0)
		total_pair_selections = perc_of_total_pairs_to_estimate * intermediate_neurons.size() * (intermediate_neurons.
			size() - 1) / 2;
	while (random_feature_correlations.size() < total_pair_selections)
	{
		int i = index_sampler(second_mt);
		int j = index_sampler(second_mt);
		// ensure ordering of idxes in pairs so we dont have to check both
		if (i == j)
			continue;
		if (i > j)
			std::swap(i, j);

		auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
		if (random_feature_correlations.contains(id_pair) || (age_restriction && (intermediate_neurons[i]->neuron_age <
			4500 || intermediate_neurons[j]->neuron_age < 4500)))
			continue;
		random_feature_correlations[id_pair] = 0;
		//random_feature_correlations[id_pair] = std::max((float)-1.0, std::min((float)1.0, get_normalized_value(i)*get_normalized_value(j)));
		random_feature_correlations_ages[id_pair] = 0;
	}
}

void SingleLayerNetwork::calculate_random_correlations(int min_estimation_period)
{
	//std::cout << "all pairs: " << std::endl;
	float alpha = 5 * (1 / float(min_estimation_period));
	// the step size here should be dependent on how long we estimate for
	//float alpha = 0.001;
	for (auto& [id_pair, corr] : random_feature_correlations)
	{
		if (random_feature_correlations_ages[id_pair] == 0)
			corr = 0;
		else
			corr = (1 - alpha) * corr + alpha * get_normalized_value(id_to_idx[id_pair.first]) * get_normalized_value(
				id_to_idx[id_pair.second]);
		random_feature_correlations_ages[id_pair]++;
	}
}

std::vector<std::pair<floatpair, std::string>> SingleLayerNetwork::replace_features_random_decorrelator_v3(
	float perc_to_replace, bool sum_features, float perc_to_decorrelate)
{
	std::vector<std::pair<intpair, float>> id_pair_correlations;
	std::vector<std::pair<floatpair, std::string>> correlated_graphviz;
	int max_replacements = int(prediction_weights.size() * perc_to_replace);
	int replaced_counter = 0;
	//int max_correlated_replacements = 2;
	int max_correlated_replacements = max(1, int(max_replacements * perc_to_decorrelate));

	// magnitude tester -> decorrelator -> magnitude tester if still needed
	while (replaced_counter < (max_replacements - max_correlated_replacements))
	{
		float least_useful_idx = min_idx(feature_utility_trace);
		// std::cout << "replacing: " << least_useful_idx << "\t util: " << feature_utility_trace[least_useful_idx] << "\t new: " << median(feature_utility_trace) << "\t non-corr replacement"<< std::endl;
		replace_features_with_idx(least_useful_idx);
		replaced_counter += 1;
	}

	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		for (int j = i + 1; j < intermediate_neurons.size(); j++)
		{
			auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
			if (feature_correlations.contains(id_pair) && fabs(random_feature_correlations[id_pair]) > 0.85 &&
				intermediate_neurons[i]->neuron_age > 4500 && intermediate_neurons[j]->neuron_age > 4500)
				id_pair_correlations.push_back(std::make_pair(id_pair, random_feature_correlations[id_pair]));
		}
	}

	if (id_pair_correlations.size())
	{
		// no sorting to get rid of maximization bias, we just randomly sample above threshold
		std::uniform_int_distribution<int> corr_index_sampler(0, id_pair_correlations.size() - 1);

		for (int c = 0; c < max_correlated_replacements; c++)
		{
			(void)c;
			auto id_pair = id_pair_correlations[corr_index_sampler(this->third_mt)].first;
			int replaced_idx;
			int i = id_to_idx[id_pair.first];
			int j = id_to_idx[id_pair.second];
			floatpair corr = std::make_pair(feature_correlations[id_pair], random_feature_correlations[id_pair]);
			//correlated_graphviz.push_back(std::make_pair(corr, get_graph(id_pair.first, id_pair.second))); //NOTE: this graph contains actual (not random) correlation est
			correlated_graphviz.push_back(std::make_pair(corr, "NA"));
			if (feature_utility_trace[i] <= feature_utility_trace[j])
			{
				if (sum_features)
					prediction_weights[j] += prediction_weights[i]; //TODO assume single outgoing w
				// std::cout << "replacing: " << i << "\t util: " << feature_utility_trace[i] << "\t new: " << median(feature_utility_trace) << "\t rcorr:" << feature_correlations[id_pair] << "\t corr: " << random_feature_correlations[id_pair] << "\t est:" << random_feature_correlations_ages[id_pair] << std::endl;
				replace_features_with_idx(i);
				replaced_idx = id_pair.first;
			}
			else
			{
				if (sum_features)
					prediction_weights[i] += prediction_weights[j];
				// std::cout << "replacing: " << j << "\t util: " << feature_utility_trace[i] << "\t new: " << median(feature_utility_trace) << "\t rcorr:" << feature_correlations[id_pair] << "\t corr: " << random_feature_correlations[id_pair] << "\t est:" << random_feature_correlations_ages[id_pair] << std::endl;
				replace_features_with_idx(j);
				replaced_idx = id_pair.second;
			}
			replaced_counter += 1;
			//if (replaced_counter >= max_correlated_replacements)
			//	break;
			auto it = std::remove_if(id_pair_correlations.begin(),
			                         id_pair_correlations.end(),
			                         [&replaced_idx](const std::pair<intpair, float>& a)
			                         {
				                         return a.first.first == replaced_idx || a.first.second == replaced_idx;
			                         });
			id_pair_correlations.erase(it, id_pair_correlations.end());
			if (id_pair_correlations.size() == 0)
				break;
		}
	}
	while (replaced_counter < max_replacements)
	{
		float least_useful_idx = min_idx(feature_utility_trace);
		// std::cout << "replacing: " << least_useful_idx << "\t util: " << feature_utility_trace[least_useful_idx] << "\t new: " << median(feature_utility_trace) << "\t non-corr replacement"<< std::endl;
		replace_features_with_idx(least_useful_idx);
		replaced_counter += 1;
	}
	return correlated_graphviz;
}


std::vector<std::pair<floatpair, std::string>> SingleLayerNetwork::replace_features_n2_decorrelator_v3(
	float perc_to_replace, bool sum_features, float perc_to_decorrelate, bool use_generate_and_test)
{
	std::vector<std::pair<intpair, float>> id_pair_correlations;
	std::vector<std::pair<floatpair, std::string>> correlated_graphviz;
	int max_replacements = int(prediction_weights.size() * perc_to_replace);
	int replaced_counter = 0;
	//int max_correlated_replacements = 2;
	int max_correlated_replacements = max(1, int(max_replacements * perc_to_decorrelate));
	// std::cout << max_correlated_replacements << " <- max corr reps " << std::endl;

	// magnitude tester -> decorrelator -> magnitude tester if still needed
	if (use_generate_and_test)
	{
		while (replaced_counter < (max_replacements - max_correlated_replacements))
		{
			float least_useful_idx = min_idx(feature_utility_trace);
			// std::cout << "replacing: " << least_useful_idx << "\t util: " << feature_utility_trace[least_useful_idx] << "\t new: " << median(feature_utility_trace) << "\t non-corr replacement"<< std::endl;
			replace_features_with_idx(least_useful_idx);
			replaced_counter += 1;
		}
	}

	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		for (int j = i + 1; j < intermediate_neurons.size(); j++)
		{
			auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
			if (feature_correlations.contains(id_pair) && fabs(feature_correlations[id_pair]) > 0.85 &&
				intermediate_neurons[i]->neuron_age > 4500 && intermediate_neurons[j]->neuron_age > 4500)
				id_pair_correlations.push_back(std::make_pair(id_pair, feature_correlations[id_pair]));
		}
	}

	if (id_pair_correlations.size())
	{
		// no sorting to get rid of maximization bias, we just randomly sample above threshold
		std::uniform_int_distribution<int> corr_index_sampler(0, id_pair_correlations.size() - 1);
		for (int c = 0; c < max_correlated_replacements; c++)
		{
			(void)c;
			auto id_pair = id_pair_correlations[corr_index_sampler(this->third_mt)].first;
			int replaced_idx;
			int i = id_to_idx[id_pair.first];
			int j = id_to_idx[id_pair.second];
			floatpair corr = std::make_pair(feature_correlations[id_pair], feature_correlations[id_pair]);
			//correlated_graphviz.push_back(std::make_pair(corr, get_graph(id_pair.first, id_pair.second)));
			correlated_graphviz.push_back(std::make_pair(corr, "NA"));
			if (feature_utility_trace[i] <= feature_utility_trace[j])
			{
				if (sum_features)
					prediction_weights[j] += prediction_weights[i]; //TODO assume single outgoing w
				// std::cout << "replacing: " << i << "\t util: " << feature_utility_trace[i] << "\t new: " << median(feature_utility_trace) << "\t corr: " << feature_correlations[id_pair] << std::endl;
				replace_features_with_idx(i);
				replaced_idx = id_pair.first;
			}
			else
			{
				if (sum_features)
					prediction_weights[i] += prediction_weights[j];
				// std::cout << "replacing: " << j << "\t util: " << feature_utility_trace[i] << "\t new: " << median(feature_utility_trace) << "\t corr: " << feature_correlations[id_pair] << std::endl;
				replace_features_with_idx(j);
				replaced_idx = id_pair.second;
			}
			replaced_counter += 1;
			//if (replaced_counter >= max_correlated_replacements)
			//	break;
			auto it = std::remove_if(id_pair_correlations.begin(),
			                         id_pair_correlations.end(),
			                         [&replaced_idx](const std::pair<intpair, float>& a)
			                         {
				                         return a.first.first == replaced_idx || a.first.second == replaced_idx;
			                         });
			id_pair_correlations.erase(it, id_pair_correlations.end());
			if (id_pair_correlations.size() == 0)
				break;
		}
	}
	// std::cout << replaced_counter << std::endl;
	if (use_generate_and_test)
	{
		while (replaced_counter < max_replacements)
		{
			float least_useful_idx = min_idx(feature_utility_trace);
			// std::cout << "replacing: " << least_useful_idx << "\t util: " << feature_utility_trace[least_useful_idx] << "\t new: " << median(feature_utility_trace) << "\t non-corr replacement"<< std::endl;
			replace_features_with_idx(least_useful_idx);
			replaced_counter += 1;
		}
	}
	return correlated_graphviz;
}


void SingleLayerNetwork::replace_features(float perc_to_replace)
{
	int max_replacements = int(prediction_weights.size() * perc_to_replace);
	int replaced_counter = 0;

	while (replaced_counter < max_replacements)
	{
		float least_useful_idx = min_idx(feature_utility_trace);
		replace_features_with_idx(least_useful_idx);
		replaced_counter += 1;
	}
}


void SingleLayerNetwork::replace_features_randomly(float perc_to_replace)
{
	std::uniform_int_distribution<int> index_sampler(0, prediction_weights.size() - 1);
	int max_replacements = int(prediction_weights.size() * perc_to_replace);
	int replaced_counter = 0;

	while (replaced_counter < max_replacements)
	{
		float least_useful_idx = min_idx(feature_utility_trace);
		int replacement_idx = index_sampler(this->mt);
		replace_features_with_idx(replacement_idx);
		replaced_counter += 1;
	}
}


float SingleLayerNetwork::forward(std::vector<float> inputs)
{
	//TODO remove initial non-stationarity due to normalization in target network?
	// Set input neurons value
	// inputs can be larger than input_neurons.size(). we simply ignore the rest
	for (int i = 0; i < input_neurons.size(); i++)
	{
		this->input_neurons[i]->value = inputs[i];
	}

	for (auto& n : this->intermediate_neurons)
		n->update_value(0);

	for (auto& n : this->intermediate_neurons)
		n->fire(0);

	for (int counter = 0; counter < intermediate_neurons.size(); counter++)
	{
		feature_mean[counter] = feature_mean[counter] * 0.9999 + 0.0001 * intermediate_neurons[counter]->value;
		if (std::isnan(feature_mean[counter]))
		{
			std::cout << "feature value = " << intermediate_neurons[counter]->value << std::endl;
			std::cout << "Exitting in single_layer_network.cpp" << std::endl;
			exit(1);
		}
		float temp = (feature_mean[counter] - intermediate_neurons[counter]->value);
		feature_std[counter] = feature_std[counter] * 0.9999 + 0.0001 * temp * temp;
		if (feature_std[counter] < this->std_cap)
			feature_std[counter] = this->std_cap;
	}

	this->v_old = this->predictions;
	predictions = 0;
	for (int i = 0; i < prediction_weights.size(); i++)
	{
		float feature_output = prediction_weights[i] * (this->intermediate_neurons[i]->value - feature_mean[i]) / sqrt(
			feature_std[i]);
		feature_utility_trace[i] = feature_utility_trace[i] * 0.9999 + 0.0001 * fabs(feature_output);
		predictions += feature_output;
	}
	predictions += bias;
	return predictions;
}


void SingleLayerNetwork::print_all_correlations()
{
	// std::cout << "feature correlations:" << std::endl;
	int count = 0;
	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		for (int j = i + 1; j < intermediate_neurons.size(); j++)
		{
			auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
			if (fabs(feature_correlations[id_pair]) > 0.85 && intermediate_neurons[i]->neuron_age > 5000 &&
				intermediate_neurons[j]->neuron_age > 5000)
			{
				std::cout << "--->\t";
				std::cout << intermediate_neurons[i]->id << "\t->\t" << intermediate_neurons[j]->id << "\t:\t" <<
					feature_correlations[id_pair] << std::endl;
				count += 1;
			}
		}
	}
	// std::cout << "total correlated: " << count << std::endl;
}


int SingleLayerNetwork::count_mature_features()
{
	int counter = 0;
	for (const auto& feature : intermediate_neurons)
	{
		if (feature->neuron_age > 100000)
			counter++;
	}
	return counter;
}


int SingleLayerNetwork::count_highly_correlated_features()
{
	//std::cout << "highly correlated idds: " << std::endl;
	std::set<int> correlated_feature_ids;
	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		for (int j = i + 1; j < intermediate_neurons.size(); j++)
		{
			auto id_pair = std::make_pair(intermediate_neurons[i]->id, intermediate_neurons[j]->id);
			if (intermediate_neurons[i]->neuron_age > 100000 && intermediate_neurons[j]->neuron_age > 100000 && fabs(
				feature_correlations[id_pair]) > 0.85)
			{
				correlated_feature_ids.insert(intermediate_neurons[i]->id);
				correlated_feature_ids.insert(intermediate_neurons[j]->id);
				//std::cout << i << "->" << j << ":" << feature_correlations[id_pair] << std::endl;
			}
		}
	}
	return correlated_feature_ids.size();
}


void SingleLayerNetwork::print_all_statistics()
{
	for (int i = 0; i < intermediate_neurons.size(); i++)
	{
		std::cout << intermediate_neurons[i]->id << "\tmean: " << feature_mean[i];
		std::cout << "\tstd: " << feature_std[i] << std::endl;
	}
}


void SingleLayerNetwork::zero_grad()
{
	//for (int counter = 0; counter < intermediate_neurons.size(); counter++) {
	//  intermediate_neurons[counter]->zero_grad();
	//}

	for (int index = 0; index < intermediate_neurons.size(); index++)
	{
		prediction_weights_gradient[index] = 0;
	}
	bias_gradients = 0;
}


std::vector<float> SingleLayerNetwork::get_feature_utilities()
{
	std::vector<float> my_vec;
	my_vec.reserve(feature_utility_trace.size());
	for (int index = 0; index < feature_utility_trace.size(); index++)
	{
		my_vec.push_back(feature_utility_trace[index]);
	}
	return my_vec;
}


std::vector<float> SingleLayerNetwork::get_prediction_weights()
{
	std::vector<float> my_vec;
	my_vec.reserve(prediction_weights.size());
	for (int index = 0; index < prediction_weights.size(); index++)
	{
		my_vec.push_back(prediction_weights[index]);
	}
	return my_vec;
}


std::vector<std::pair<float, int>> SingleLayerNetwork::get_prediction_weight_statistics()
{
	std::vector<std::pair<float, int>> my_vec;
	my_vec.reserve(prediction_weights.size());
	for (int index = 0; index < prediction_weights.size(); index++)
	{
		auto new_pair = std::make_pair(prediction_weights[index], intermediate_neurons[index]->neuron_age);
		my_vec.push_back(new_pair);
	}
	return my_vec;
}


std::vector<float> SingleLayerNetwork::get_prediction_gradients()
{
	std::vector<float> my_vec;
	my_vec.reserve(prediction_weights_gradient.size());
	for (int index = 0; index < prediction_weights_gradient.size(); index++)
	{
		my_vec.push_back(prediction_weights_gradient[index]);
	}
	return my_vec;
}


void SingleLayerNetwork::backward()
{
	for (int index = 0; index < prediction_weights.size(); index++)
		prediction_weights_gradient[index] += (intermediate_neurons[index]->value - feature_mean[index]) /
			sqrt(feature_std[index]);
	bias_gradients += 1;
}

void SingleLayerNetwork::decayGradients(float decay_rate)
{
	for (int index = 0; index < prediction_weights.size(); index++)
		prediction_weights_gradient[index] *= decay_rate;
	bias_gradients *= decay_rate;
}


void SingleLayerNetwork::update_parameters_only_prediction(float error)
{
	//TODO assumes single outgoing weight from interm neuron

	for (int index = 0; index < prediction_weights.size(); index++)
		prediction_weights[index] += prediction_weights_gradient[index] * error * step_size;
	bias += error * step_size * bias_gradients;
}


void SingleLayerNetwork::update_parameters_only_prediction(float error, float l2_lambda, float l1_lambda)
{
	//TODO assumes single outgoing weight from interm neuron
	for (int index = 0; index < prediction_weights.size(); index++)
		//prediction_weights[index] += step_size * (error * prediction_weights_gradient[index] - l2_lambda * prediction_weights[index] - l1_lambda * (fabs(prediction_weights[index])/prediction_weights[index]));
		prediction_weights[index] += step_size * (error * prediction_weights_gradient[index] - l2_lambda *
			prediction_weights[index]);
}


void SingleLayerNetwork::update_parameters_only_prediction_RMSProp(float error)
{
	float eps = 1e-8;
	float a = 0.99;
	for (int i = 0; i < prediction_weights.size(); i++)
	{
		float grad = error * prediction_weights_gradient[i];
		// compute trace of squared gradient
		prediction_weights_gradient_trace[i] = a * prediction_weights_gradient_trace[i] + (1 - a) * pow(grad, 2);
		// update using the RMSProp optimizer
		prediction_weights[i] += step_size * grad / sqrt(prediction_weights_gradient_trace[i] + eps);
	}
}


void SingleLayerNetwork::update_parameters(float error)
{
	//TODO assumes single outgoing weight from interm neuron
	for (int index = 0; index < intermediate_neurons.size(); index++)
	{
		float incoming_gradient = error * (prediction_weights[index] / sqrt(feature_std[index])) * intermediate_neurons[
			index]->backward(intermediate_neurons[index]->value);
		for (const auto& synapse : intermediate_neurons[index]->incoming_synapses)
			synapse->weight += step_size * synapse->input_neuron->value * incoming_gradient;
	}
	for (int index = 0; index < prediction_weights.size(); index++)
		prediction_weights[index] += prediction_weights_gradient[index] * error * step_size;
	//  bias += error * step_size * 0.001 * bias_gradients;
}


float SingleLayerNetwork::read_output_values()
{
	return predictions;
}


SingleLayerNetwork::~SingleLayerNetwork()
{
};
