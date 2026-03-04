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
    std::vector<std::vector<std::string>> csv = loadCSV("../Examples/Datasets/Iris.csv");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(csv.begin(), csv.end(), gen);
    std::map<std::string, std::vector<int>> class_onehot = {
        {"Iris-setosa", {1, 0, 0}},
        {"Iris-versicolor", {0, 1, 0}},
        {"Iris-virginica", {0, 0, 1}}
    };

    std::string device = "cuda";

    int features = 4;
    int rows = csv.size();
    int classes = 3;

    Tensor_ptr df = Tensor::init({rows, features}, true, device);
    Tensor_ptr target = Tensor::init({rows, classes}, true, device);
    for (int i = 0; i < csv.size(); ++i) {
        for (int j = 1; j < features; ++j) {
            df->at({i, j-1}) = std::stod(csv[i][j]); // ignore id
        }
        std::vector<int> onehot = class_onehot[csv[i][features+1]];
        for (int j = 0; j < onehot.size(); ++j) {
            target->at({i,j}) = onehot[j];
        }
    }

    // split train test
    int train = rows * 0.8, test = rows - train;
    Tensor_ptr df_train = Tensor::init({train, features}, true, device);
    Tensor_ptr target_train = Tensor::init({train, classes}, true, device);
    Tensor_ptr df_test = Tensor::init({test, features}, true, device);
    Tensor_ptr target_test = Tensor::init({test, classes}, true, device);

    for (int i = 0; i < train; ++i) {
        for (int j = 0; j < features; ++j) {
            df_train->at({i, j}) = df->at({i, j});
        }
        for (int j = 0; j < classes; ++j) {
            target_train->at({i,j}) = target->at({i,j});
        }
    }

    for (int i = 0; i < test; ++i) {
        for (int j = 0; j < features; ++j) {
            df_test->at({i,j}) = df->at({i + train, j});
        }
        for (int j = 0; j < classes; ++j) {
            target_test->at({i,j}) = target->at({i + train, j});
        }
    }

    printf("train size: %i, test size: %i\n", df_train->get_shape(0), df_test->get_shape(0));

    // model
    Linear l1(4, 16, device);
    Linear l2(16, 3, device);
    SGD optim({l1.params(), l2.params()}, 0.01);

    // train
    printf("TRAIN\n");
    int epochs = 1000;
    for (int i = 0; i < epochs; ++i)
    {
        optim.zero_grad();
        Tensor_ptr x = l1.forward(df_train)->relu();
        Tensor_ptr x2 = l2.forward(x);
        Tensor_ptr loss = MSELoss(x2, target_train);
        loss->backward();
        optim.step();
        printf("epoch: %i, loss: %f\n", i, loss->at(0));
    }

    // test
    printf("TEST\n");
    Tensor_ptr result = l1.forward(df_test)->relu();
    result = l2.forward(result);
    result = result->argmax(1);

    int correct = 0;
    for (int i = 0; i < target_test->get_shape(0); ++i)
    {
        int result_idx = result->at({i, 0}), target_idx = -1;
        printf("pred: %i, ", result_idx);
        for (int j = 0; j < classes; ++j)
        {
            if (target_test->at({i,j}) == 1.0) {
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