//
// Created by Khurram Javed on 2024-11-04.
//

#ifndef PREDICTIONEVALUATOR_H
#define PREDICTIONEVALUATOR_H

#include <iostream>
#include <vector>

class PredictionEvaluator
{
    std::vector<float> rewards;
    std::vector<float> predictions;
    std::vector<float> returns;
    float gamma;
    void calculateReturns();

public:
    int getAge();
    PredictionEvaluator(float gamma);
    void addPredictionAndReward(float prediction, float reward);
    float getMSE(int lifetime);
    float getMSEOverAllExperience();
    std::vector<std::pair<int, float>> getPredictions(int t1, int t2);
    std::vector<std::pair<int, float>> getReturns(int t1, int t2);
    std::vector<std::pair<int, float>> getErrorOverLifetime(int lifetime, int n);
    std::vector<std::pair<int, float>> getRewards(int t1, int t2);
};

#endif //PREDICTIONEVALUATOR_H
