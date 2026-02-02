#include "examples_helpers.hpp"
#include "tensor.hpp"
#include "linear.hpp"
#include "optim.hpp"
#include "loss.hpp"
#include <iostream>
#include <map>
#include <random>

#include <chrono>
#include <ctime>

int main()
{
    std::time_t currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Current time: " << std::ctime(&currentTime);
    
    // make df
    std::vector<std::vector<std::string>> csv = loadCSV("Examples/Datasets/Iris.csv");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(csv.begin(), csv.end(), gen);
    std::map<std::string, std::vector<int>> class_onehot = {
        {"Iris-setosa", {1, 0, 0}},
        {"Iris-versicolor", {0, 1, 0}},
        {"Iris-virginica", {0, 0, 1}}
    };

    int features = 4;
    int rows = csv.size();
    int classes = 3;

    Tensor df({rows, features});
    Tensor target({rows, classes});
    for (int i = 0; i < csv.size(); ++i) {
        for (int j = 1; j < features; ++j) {
            df[i][j-1] = std::stod(csv[i][j]); // ignore id
        }
        std::vector<int> onehot = class_onehot[csv[i][features+1]];
        for (int j = 0; j < onehot.size(); ++j) {
            target[i][j] = onehot[j];
        }
    }

    // split train test
    int train = rows * 0.8, test = rows - train;
    Tensor df_train({train, features});
    Tensor target_train({train, classes});
    Tensor df_test({test, features});
    Tensor target_test({test, classes});

    for (int i = 0; i < train; ++i) {
        for (int j = 0; j < features; ++j) {
            df_train[i][j] = df[i][j]->data;
        }
        for (int j = 0; j < classes; ++j) {
            target_train[i][j] = target[i][j]->data;
        }
    }

    for (int i = 0; i < test; ++i) {
        for (int j = 0; j < features; ++j) {
            df_test[i][j] = df[i + train][j]->data;
        }
        for (int j = 0; j < classes; ++j) {
            target_test[i][j] = target[i + train][j]->data;
        }
    }

    printf("train size: %i, test size: %i\n", df_train.shape[0], df_test.shape[0]);

    // model
    Linear l1(4, 16);
    Linear l2(16, 3);
    SGD optim({l1.params(), l2.params()}, 0.01);

    // train
    printf("TRAIN\n");
    int epochs = 300;
    for (int i = 0; i < epochs; ++i)
    {
        optim.zero_grad();
        Tensor x = l1.forward(df_train).relu();
        x = l2.forward(x);
        Value_ptr loss = MSELoss(x, target_train);
        loss->backward();
        optim.step();
        printf("epoch: %i, loss: %f\n", i, loss->data);
    }

    // test
    printf("TEST\n");
    Tensor result = l1.forward(df_test).relu();
    result = l2.forward(result);
    result = result.argmax(1);

    int correct = 0;
    for (int i = 0; i < target_test.shape[0]; ++i)
    {
        int result_idx = result[i][0].get()->data, target_idx = -1;
        printf("pred: %i, ", result_idx);
        for (int j = 0; j < classes; ++j)
        {
            if (target_test[i][j].get()->data == 1.0) {
                target_idx = j;
                printf("target: %i", j);
            }
        }
        if (target_idx == -1) printf("TARGET ERROR");
        if (result_idx == -1) printf("RESULT ERROR");
        printf("\n");
        
        correct += result_idx == target_idx;
    }

    printf("correct: %i\n", correct);
    printf("correct% : %f\n", float(correct)/float(test));
}