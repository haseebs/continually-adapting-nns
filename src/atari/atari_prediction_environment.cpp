//
// Created by Khurram Javed on 2022-07-19.
//

#include "../../include/environments/atari_prediction_environment.h"
#include <fstream>
#include <iostream>

AtariPredictionEnvironment::AtariPredictionEnvironment(std::string game_name,
                                                       float gamma, std::string policy_type)
    : gray_features(210 * 160, 0), observation(84 * 84 * 16 + 20, 0), generator(0)
{
    //    std::cout << "Initializing env\n";\\

    std::fstream* policy = new std::fstream("../policies/" + game_name + "NoFrameskip-v4.txt",
                                            std::ios::in | std::ios::binary);
    //    std::cout << "Policy loaded\n";
    my_env = new ale::ALEInterface();
    this->gamma = gamma;
    this->policy_type = policy_type;

    my_env->setInt("random_seed", 1731038949);
    my_env->setFloat("repeat_action_probability", 0.0);
    my_env->setInt("frame_skip", 1);
    my_env->loadROM("../games/" + game_name + ".bin");
    my_env->reset_game();

    long size;
    policy->seekg(0, std::ios::end);
    size = policy->tellg();
    policy->seekg(0, std::ios::beg);
    actions = new char[size];
    policy->read(actions, size);
    policy->close();
    delete policy;
    action_set = my_env->getMinimalActionSet();
    time = 0;
    reward = 0;
    ep_reward = 0;
    to_reset = false;
    this->last_action = 0;
    this->to_step = true;
    this->alive = true;
}

AtariPredictionEnvironment::~AtariPredictionEnvironment()
{
}


std::vector<unsigned int> AtariPredictionEnvironment::get_state()
{
    return {};
}


std::vector<unsigned int> AtariPredictionEnvironment::step()
{
    to_reset = false;
    time++;
    if (actions[time] == 'R')
    {
        reward = 0;
        my_env->reset_game();
        to_reset = true;
    }
    else
    {
        auto a = my_env->getMinimalActionSet();
        reward = my_env->act(my_env->getMinimalActionSet()[int(actions[time]) - 97]);
    }
    return this->get_state();
}


float AtariPredictionEnvironment::get_target() { return real_target[time]; }

float AtariPredictionEnvironment::get_gamma()
{
    return this->gamma;
}

float AtariPredictionEnvironment::get_reward()
{
    if (reward > 0.1)
        return 1;
    if (reward < -0.1)
        return -1;
    return 0;
}
