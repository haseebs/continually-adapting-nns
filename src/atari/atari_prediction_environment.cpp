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
    //  my_env.setBool("truncate_on_loss_of_life", true);
    my_env->setFloat("repeat_action_probability", 0.0);
    my_env->setInt("frame_skip", 1);
    //    std::cout << "Int set\n";
    my_env->loadROM("../games/" + game_name + ".bin");
    //    std::cout << "Bin loaded\n";
    my_env->reset_game();

    long size;
    policy->seekg(0, std::ios::end);
    size = policy->tellg();
    policy->seekg(0, std::ios::beg);
    actions = new char[size];
    policy->read(actions, size);
    policy->close();
    delete policy;
    //    std::cout << "Size of actions = " << size << std::endl;
    action_set = my_env->getMinimalActionSet();
    std::cout << "Minimal action set size = " << action_set.size() << std::endl;
    time = 0;
    reward = 0;
    ep_reward = 0;
    to_reset = false;
    std::cout << "Constructor done\n";
    this->last_action = 0;
    this->to_step = true;
    this->alive = true;
    this->pinned_memory = new unsigned char[210 * 160 * 3 + 2 * sizeof(float)];
    this->ram_memory = new unsigned char[128 + 2 * sizeof(float)];

}

AtariPredictionEnvironment::~AtariPredictionEnvironment()
{
    delete pinned_memory;
    delete my_env;
}


std::vector<unsigned int> AtariPredictionEnvironment::get_state()
{
    return {};
}

void AtariPredictionEnvironment::UpdateReturns()
{
    float old_val = 0;
    list_of_returns = std::vector<float>(list_of_rewards.size(), 0);
    for (int i = list_of_rewards.size() - 1; i >= 0; i--)
    {
        list_of_returns[i] = list_of_rewards[i] + old_val;
        old_val = list_of_returns[i] * this->gamma;
    }
    list_of_rewards.clear();
}

// S, 1, S, 1, S, R,
std::vector<float>& AtariPredictionEnvironment::GetListOfReturns()
{
    return this->list_of_returns;
}

bool AtariPredictionEnvironment::get_done() { return true; }

std::vector<float> FastStep()
{
    return {};
}

std::vector<unsigned int> AtariPredictionEnvironment::step()
{
    to_reset = false;
    time++;
    if (actions[time] == 'R')
    {
        std::cout << "Resetting\n";
        reward = 0;
        my_env->reset_game();
        to_reset = true;
    }
    else
    {
        this->last_action = int(actions[time]) - 97;
        reward = my_env->act(action_set[int(actions[time]) - 97]);
    }
    return this->get_state();
}


float AtariPredictionEnvironment::get_target() { return real_target[time]; }

float AtariPredictionEnvironment::get_gamma()
{
    //    if (to_reset)
    //        return 0;
    return this->gamma;
}

float AtariPredictionEnvironment::get_reward()
{
    if (reward > 0.1)
        return 1;
    else if (reward < -0.1)
        return -1;
    return 0;
}
