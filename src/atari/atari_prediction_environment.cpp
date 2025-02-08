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

void AtariPredictionEnvironment::step_loop()
{
    while (alive)
    {
        if (to_step)
        {
            if (policy_type == "pretrained")
            {
                // std::cout << "pretrained policy\n";
                step();
            }
            else if (policy_type == "random")
            {
                auto actions = my_env->getMinimalActionSet();
                std::uniform_int_distribution<int> dist(0, actions.size() - 1);
                current_action = dist(generator);
                TakeAction(current_action);
            }
            else if(policy_type == "custom")
            {
                float cumulative_reward = 0;

                TakeAction(current_action);
//                action_queue.push(current_action);
//
//                for(int i = 0; i < 4; i++){
//                    TakeAction(current_action);
//                    cumulative_reward += reward;
//                }
//                reward = cumulative_reward;
            }
            my_env->getScreenRGB(gray_features);
            std::memcpy(pinned_memory, gray_features.data(), gray_features.size());
            auto r = my_env->getRAM();
            auto* ptr = r.array();
            float reward = this->get_reward();
            float gamma = this->get_gamma();
            memcpy(ram_memory, ptr, 128);
            std::memcpy(ram_memory + 128, &reward, sizeof(float));
            std::memcpy(ram_memory + 128 + sizeof(float), &gamma, sizeof(float));



            std::memcpy(pinned_memory + gray_features.size(), &reward, sizeof(float));
            std::memcpy(pinned_memory + gray_features.size() + sizeof(float), &gamma, sizeof(float));
            to_step = false;
        }
        else
        {
            // std::cout << "Skipping step\n";
        }
    }
}

void AtariPredictionEnvironment::async_step()
{
    step_thread = std::thread(&AtariPredictionEnvironment::step_loop, this);
}

void AtariPredictionEnvironment::wait() { step_thread.join(); }

float AtariPredictionEnvironment::TakeAction(int action)
{
    this->last_action = current_action;
    current_action = action;
    time++;
    my_env->getMinimalActionSet();
    // print Minimal Action Set
//     std::cout << "Minimal Action Set: ";
//     for (int i = 0; i < my_env->getMinimalActionSet().size(); i++)
//     {
//         std::cout << my_env->getMinimalActionSet()[i] << " ";
//     }
//     std::cout << std::endl;
    reward = my_env->act(my_env->getMinimalActionSet()[action]);
    if (my_env->game_over())
    {
        my_env->reset_game();
    }
    if (reward > 0.1)
    {
        return 1;
    }
    else if (reward < -0.1)
    {
        return -1;
    }
    return 0;
}


std::vector<float> AtariPredictionEnvironment::FastStep()
{
    to_reset = false;
    time++;

    if (actions[time] == 'R')
    {
        my_env->reset_game();
        to_reset = true;
    }
    else
    {
        reward = my_env->act(action_set[int(actions[time]) - 97]);
    }

    return {};
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
