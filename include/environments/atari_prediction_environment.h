//
// Created by Khurram Javed on 2022-07-19.
//

#ifndef INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_
#define INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_

#include "../../src/atari/ale_interface.hpp"
#include "../../src/atari/common/Constants.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <queue>

class AtariPredictionEnvironment
{
protected:
    std::mt19937 generator;
    std::vector<unsigned char> gray_features;
    std::vector<float> observation;

    bool to_reset;
    std::vector<float> list_of_rewards;
    std::vector<float> list_of_returns;
    std::vector<ale::Action> action_set;
    char* actions;

    float ep_reward;

    std::atomic<bool> alive;
    std::thread step_thread;

    void UpdateReturns();

public:
    float reward;
    int last_action;
    int current_action;
    ale::ALEInterface* my_env;
    std::vector<float> real_target;
    int time;
    int total;
    std::atomic<bool> to_step;
    float gamma;
    std::queue<int> action_queue;

    int old_action;

    std::string policy_type;

    AtariPredictionEnvironment(std::string path, float gamma, std::string policy_type);

    ~AtariPredictionEnvironment();

    std::vector<float>& GetListOfReturns();

    std::vector<unsigned int> get_state();

    std::vector<unsigned int> step();

    float get_target();

    float get_gamma();


    bool get_done();

    float get_reward();
};
#endif // INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_
