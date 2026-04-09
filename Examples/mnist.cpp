// WORK IN PROGRESS

#include "examples_helpers.hpp"
#include "tensor.hpp"
#include "linear.hpp"
#include "optim.hpp"
#include "loss.hpp"
#include "globals.hpp"
#include <iostream>
#include <map>
#include <random>

#include <chrono>
#include <ctime>

void print_vec(std::vector<int> vec) {
    for (auto& v : vec)
        printf("%i ", v);
    printf("\n");
}

int main()
{
    std::time_t currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "START: " << std::ctime(&currentTime);
    
    // make df
    std::vector<std::vector<std::string>> train_csv = loadCSV("/home/pawel/cppmlbib/Examples/Datasets/mnist_test.csv");
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::shuffle(train_csv.begin(), train_csv.end(), gen);
    
    int train_rows = 100;
    int test_rows = 100;
    // int train_rows = train_csv.size();
    // int test_rows = test_csv.size();
    
    Device device = Device::CUDA;
    int size = 28;
    int classes = 10;
    int kernel_size = 4;
    int stride = 2;
    int padding = 0;
    int batch_size = 64;
    int input_channels = 1;

    int minibatches = (train_rows-1)/batch_size + 1;
    std::vector<Tensor_ptr> train_dl(minibatches);
    std::vector<Tensor_ptr> target_train_dl(minibatches);


    // ---- TRAIN dl
    int it = 0;
    for (int b = 1; b < train_rows; b += batch_size, ++it) {
        if (batch_size >= train_rows - b) break;
        Tensor_ptr values = Tensor::init({batch_size, size, size, input_channels}, true, device);
        Tensor_ptr target = Tensor::init({batch_size, classes}, true, device);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < size * size; ++j) {
                values->set({i, j / size, j % size, 0}, std::stod(train_csv[b+i][j+1]) / 255);
            }

            int target_id = std::stoi(train_csv[b+i][0]);
            target->set({i, target_id}, 1);
        }
        train_dl[it] = values->im2col(kernel_size, stride, padding);
        target_train_dl[it] = target;
    }
    minibatches = std::min(it, minibatches); // TODO: fix
    train_dl.resize(minibatches);
    target_train_dl.resize(minibatches);


    currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "PREPARED DF: " << std::ctime(&currentTime);
    printf("train size: %i (%i batches), test size: %i\n", train_rows-1, minibatches, test_rows-1);

    // model
    Convolution c1(1, 32, kernel_size, stride, padding, device);
    Convolution c2(32, 16, kernel_size, stride, padding, device);
    Linear l3(400, 128, device);
    Linear l4(128, 10, device);
    SGD optim({c1.params(), c2.params(), l3.params(), l4.params()}, 0.01, device);

    // train
    printf("TRAIN\n");
    int epochs = 5;
    for (int i = 0; i < epochs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        float loss_sum = 0;
        for (int b = 0; b < minibatches; ++b) {
            // auto batch_start = std::chrono::high_resolution_clock::now();
            optim.zero_grad();
            Tensor_ptr x = c1.forward(train_dl[b])->relu();
            Tensor_ptr x2 = c2.forward(x)->relu();
            x2 = x2->view({batch_size, x2->get_shape(1) * x2->get_shape(2) * x2->get_shape(3)});
            Tensor_ptr x3 = l3.forward(x2)->relu();
            Tensor_ptr x4 = l4.forward(x3)->softmax();
            Tensor_ptr loss = MSELoss(x4, target_train_dl[b]);
            // auto forward_end = std::chrono::high_resolution_clock::now();
            loss->backward();
            // auto backward_end = std::chrono::high_resolution_clock::now();
            optim.step();
            loss_sum += loss->get(0);
            // std::chrono::duration<double> elapsed_forward = (forward_end - batch_start);
            // std::chrono::duration<double> elapsed_backward = (backward_end - forward_end);
            // printf("forward: %f s, backward %f s\n", elapsed_forward.count(), elapsed_backward.count());
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        printf("epoch: %i, loss: %f, time: %f sec\n", i, loss_sum, elapsed.count());
    }

    // test
    // printf("TEST\n");
    //     std::vector<std::vector<std::string>> test_csv = loadCSV("/home/pawel/cppmlbib/Examples/Datasets/mnist_test.csv");

    // // ----- TEST df
    // Tensor_ptr df_test = Tensor::init({test_rows-1, size, size, input_channels}, true, device);
    // Tensor_ptr target_test = Tensor::init({test_rows-1, classes}, true, device);
    // for (int i = 1; i < test_rows; ++i) {
    //     for (int j = 0; j < size * size; ++j) {
    //         df_test->set({i-1, j / size, j % size, 0}, std::stod(test_csv[i][j+1]) / 255);
    //     }
    //     int target_id = std::stoi(test_csv[i][0]);
    //     target_test->set({i-1, target_id}, 1);
    // }

    // Tensor_ptr x = c1.forward(df_test)->relu();
    // Tensor_ptr x2 = c2.forward(x)->relu();
    // x2 = x2->view({test_rows-1, x2->get_shape(1) * x2->get_shape(2) * x2->get_shape(3)});
    // Tensor_ptr x3 = l3.forward(x2)->relu();
    // Tensor_ptr x4 = l4.forward(x3)->softmax();
    // Tensor_ptr result = x4->argmax(1);

    // int correct = 0;
    // for (int i = 0; i < target_test->get_shape(0); ++i)
    // {
    //     int result_idx = result->get({i, 0}), target_idx = -1;
    //     printf("pred: %i, ", result_idx);
    //     for (int j = 0; j < classes; ++j)
    //     {
    //         if (target_test->get({i,j}) == 1.0) {
    //             target_idx = j;
    //             printf("target: %i", j);
    //         }
    //     }
    //     if (target_idx == -1) printf("TARGET ERROR");
    //     if (result_idx == -1) printf("RESULT ERROR");
    //     printf("\n");
        
    //     correct += result_idx == target_idx;
    // }

    // printf("correct: %i\n", correct);
    // printf("correct percentage : %f\n", float(correct)/float(test_rows));
}
