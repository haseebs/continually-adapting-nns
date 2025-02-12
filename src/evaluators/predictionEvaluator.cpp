//
// Created by Khurram Javed on 2024-11-04.
//

#include "../../include/evaluators/predictionEvaluator.h"


void PredictionEvaluator::addPredictionAndReward(float prediction, float reward)
{
    predictions.push_back(prediction);
    rewards.push_back(reward);
}

void PredictionEvaluator::calculateReturns()
{
    returns.clear();
    returns = std::vector<float>(rewards.size(), 0);
    returns[returns.size() - 1] = 0;
    double return_val = 0;
    for (int j = rewards.size() - 1; j >= 1; j--)
    {
        return_val = rewards[j] + gamma * return_val;
        returns[j - 1] = return_val;
    }
}

PredictionEvaluator::PredictionEvaluator(float gamma)
{
    this->gamma = gamma;
}

float PredictionEvaluator::getMSE(int lifetime)
{
    calculateReturns();
    long double mse = 0;
    if (lifetime > predictions.size())
    {
        std::cout << "Lifetime is greater than the number of predictions made. Exiting" << std::endl;
        exit(1);
    }
    for (int i = 0; i < lifetime; i++)
    {
        mse += (returns[i] - predictions[i]) * (returns[i] - predictions[i]);
    }
    return static_cast<float>(mse / static_cast<double>(lifetime));
}

float PredictionEvaluator::getMSEOverAllExperience()
{
    calculateReturns();
    long double mse = 0;
    int lifetime = predictions.size();
    if (lifetime > predictions.size())
    {
        std::cout << "Lifetime is greater than the number of predictions made. Exiting" << std::endl;
        exit(1);
    }
    for (int i = 0; i < lifetime; i++)
    {
        mse += (returns[i] - predictions[i]) * (returns[i] - predictions[i]);
    }
    return static_cast<float>(mse / static_cast<double>(lifetime));
}


std::vector<std::pair<int, float>> PredictionEvaluator::getErrorOverLifetime(int lifetime, int n)
{
    calculateReturns();
    std::vector<std::pair<int, float>> errors;
    if (lifetime > predictions.size())
    {
        std::cout << "Lifetime is greater than the number of predictions made. Exiting" << std::endl;
        exit(1);
    }
    int jump = lifetime / n;
    long double mse = 0;

    for (int i = 0; i < lifetime; i++)
    {
        mse += (returns[i] - predictions[i]) * (returns[i] - predictions[i]);
        if (i % jump == 0)
            errors.push_back({i, (mse / static_cast<double>(i + 1))});
    }
    return errors;
}

std::vector<std::pair<int, float>> PredictionEvaluator::getPredictions(int t1, int t2)
{
    std::vector<std::pair<int, float>> predictions;
    if (t2 > this->predictions.size())
    {
        std::cout << "t2 is greater than the number of predictions made. Exiting" << std::endl;
        exit(1);
    }
    for (int i = t1; i < t2; i++)
    {
        predictions.push_back({i, this->predictions[i]});
    }
    return predictions;
}

std::vector<std::pair<int, float>> PredictionEvaluator::getReturns(int t1, int t2)
{
    calculateReturns();
    std::vector<std::pair<int, float>> returns;
    if (t2 > this->returns.size())
    {
        std::cout << "t2 is greater than the number of predictions made. Exiting" << std::endl;
        exit(1);
    }
    for (int i = t1; i < t2; i++)
    {
        returns.push_back({i, this->returns[i]});
    }
    return returns;
}

int PredictionEvaluator::getAge()
{
    return predictions.size();
}


std::vector<std::pair<int, float>> PredictionEvaluator::getRewards(int t1, int t2)
{
    std::vector<std::pair<int, float>> rewards;
    if (t2 > this->rewards.size())
    {
        std::cout << "t2 is greater than the number of predictions made. Exiting" << std::endl;
        exit(1);
    }
    for (int i = t1; i < t2; i++)
    {
        rewards.push_back({i, this->rewards[i]});
    }
    return rewards;
}