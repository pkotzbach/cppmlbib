#include "helpers.hpp"
#include "tensor.hpp"
#include "linear.hpp"
#include "optim.hpp"
#include "loss.hpp"
#include <iostream>
#include <map>

#include <chrono>
#include <ctime>

int main()
{
    std::time_t currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Current time: " << std::ctime(&currentTime);
    // make df
    std::vector<std::vector<std::string>> csv = loadCSV("Examples/Datasets/Iris.csv");
    std::map<std::string, std::vector<int>> class_onehot = {
        {"Iris-setosa", {1, 0, 0}},
        {"Iris-versicolor", {0, 1, 0}},
        {"Iris-virginica", {0, 0, 1}}
    };

    int features = csv[0].size();
    int rows = csv.size()-1;
    int classes = 3;

    Tensor df({rows, features-1});
    Tensor target({rows, classes});
    for (int i = 1; i < csv.size(); ++i) {
        for (int j = 1; j < features - 1; ++j) {
            df[i-1][j-1] = std::stod(csv[i][j]); // ignore headers and id
        }
        std::vector<int> onehot = class_onehot[csv[i][features-1]];
        for (int j = 0; j < onehot.size(); ++j) {
            target[i-1][j] = onehot[j];
        }
    }
    currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "df: " << std::ctime(&currentTime);

    // model
    Linear l1(5, 64);
    Linear l2(64, 3);
    Softmax s;
    SGD optim({l1.params(), l2.params()}, 0.001);

    // learn

    // --------- 1 epoch
    // optim.zero_grad();
    // Value_ptr loss = std::make_shared<Value>(0);
    // // for (int j = 0; j < df.size(); ++j) {
    // int j = 0;
    // {
    //     Tensor& input = df[j].first;
    //     Tensor& target = df[j].second;
    //     Tensor x = l1.forward(input).relu();
    //     x = l2.forward(x).relu();
    //     // x = s.forward(x);
    //     loss = loss + MSELoss(x, target);
    // }

    // loss->backward();
    // optim.step();
    // printf("loss: %f\n", loss->data);
    // for (int i = 0; i < l1.weights.total_count; ++i)
    //     printf("(%f %f) ", l1.weights.values[i]->data, l1.weights.values[i]->grad);
    // for (int i = 0; i < l1.biases.total_count; ++i)
    //     printf("(%f %f) ", l1.biases.values[i]->data, l1.biases.values[i]->grad);
    // printf("\n\n");
    // for (int i = 0; i < l2.weights.total_count; ++i)
    //     printf("(%f %f) ", l2.weights.values[i]->data, l2.weights.values[i]->grad);
    // for (int i = 0; i < l2.biases.total_count; ++i)
    //         printf("(%f %f) ", l2.biases.values[i]->data, l2.biases.values[i]->grad);
    // printf("\n");


    // ---------- epochs
    int epochs = 100;
    for (int i = 0; i < epochs; ++i)
    {
        currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "epoch " << i << " " << std::ctime(&currentTime);
        optim.zero_grad();
        Tensor x = l1.forward(df).relu();
        x = l2.forward(x).relu();
        Value_ptr loss = MSELoss(x, target);
        currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "forward: " << std::ctime(&currentTime);

        loss->backward();

        currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "backward: " << std::ctime(&currentTime);
        optim.step();
        printf("epoch: %i, loss: %f\n", i, loss->data);
        // for (int i = 0; i < l2.weights.total_count; ++i)
        //     printf("(%f %f) ", l2.weights.values[i]->data, l2.weights.values[i]->grad);
        // printf("\n");
    }

}