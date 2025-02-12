#ifndef INCLUDE_NN_NETWORKS_SINGLE_LAYER_NETWORK_H_
#define INCLUDE_NN_NETWORKS_SINGLE_LAYER_NETWORK_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synced_synapse.h"
#include "../synced_neuron.h"
#include "../dynamic_elem.h"
#include "./synced_network.h"

typedef std::pair<int, int> intpair;
typedef std::pair<float, float> floatpair;

class SingleLayerNetwork
{
protected:
    int64_t time_step;
    std::mt19937 mt;
    std::mt19937 second_mt;
    std::mt19937 third_mt;

public:
    //
    //

    float step_size;
    float predictions;
    float bias;
    float bias_gradients;
    float std_cap;

    float v_old;

    std::vector<SyncedSynapse*> input_synapses;

    std::vector<float> prediction_weights;

    std::vector<float> feature_mean;

    std::vector<float> feature_std;

    std::vector<float> feature_utility_trace;

    std::vector<float> prediction_weights_gradient;
    std::vector<float> prediction_weights_gradient_trace;

    std::map<int, int> id_to_idx; // intermediate_neurons[idx]->id to idx
    std::map<intpair, float> feature_correlations;
    std::map<intpair, float> random_feature_correlations;
    std::map<intpair, int> random_feature_correlations_ages;

    float get_target_without_sideeffects(std::vector<float> inputs);

    std::vector<SyncedNeuron*> input_neurons;

    std::vector<SigmoidSyncedNeuron*> intermediate_neurons;

    float read_output_values();

    void replace_features(float perc_to_replace);
    void replace_features_randomly(float perc_to_replace);
    std::vector<std::pair<floatpair, std::string>> replace_features_n2_decorrelator_v3(
        float perc_to_replace, bool sum_features, float perc_to_decorrelate, bool use_generate_and_test);
    std::vector<std::pair<floatpair, std::string>> replace_features_random_decorrelator_v3(
        float perc_to_replace, bool sum_features, float perc_to_decorrelate);

    SingleLayerNetwork(float step_size, int seed, int no_of_input_features, int no_of_intermediate_features,
                       bool is_target_network);

    ~SingleLayerNetwork();

    float forward(std::vector<float> inputs);

    void decayGradients(float decay_rate);

    void zero_grad();

    void backward();

    void update_parameters(float error);
    void update_parameters_only_prediction(float error);
    void update_parameters_only_prediction(float error, float l2_lambda, float l1_lambda);
    void update_parameters_only_prediction_RMSProp(float error);

    std::vector<float> get_prediction_gradients();
    std::vector<float> get_prediction_weights();
    std::vector<float> get_feature_utilities();
    float get_normalized_value(int idx);

    void calculate_all_correlations();
    void print_all_correlations();
    void print_all_statistics();
    int count_highly_correlated_features();
    int count_mature_features();
    void replace_features_with_idx(int feature_idx);
    void decorrelate_features_baseline(int sum_features);
    std::vector<std::pair<float, int>> get_prediction_weight_statistics();

    void calculate_random_correlations(int min_estimation_period);
    void update_random_correlation_selections(bool age_restriction, float perc_of_total_pairs_to_estimate);
    void update_random_correlation_selections_using_thresh(bool age_restriction);
    float get_normalized_values(int idx);

    std::string get_graph(int id1, int id2);
};


#endif //INCLUDE_NN_NETWORKS_SINGLE_LAYER_NETWORK_H_
