#include "value.hpp"
#include "linear.hpp"
#include "tensor.hpp"
#include "loss.hpp"
#include "optim.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

inline void assert_almost_equal(double a, double b, double epsilon = 1e-4) {
    if (std::fabs(a - b) >= epsilon) {
        std::cerr << "Assertion failed!\n"
                  << "  a = " << a << "\n"
                  << "  b = " << b << "\n";
        throw std::runtime_error("assert_almost_equal");
    }
}

inline Tensor make_tensor(std::vector<std::vector<double>> data) {
    Tensor input({data.size(), data[0].size()});
    for (int i = 0; i < data.size(); ++i)
        for (int j = 0; j < data[i].size(); ++j)
            input[i][j] = data[i][j];
    return input;
}

inline void assert_flatten_tensor_almost_equal(Tensor& a, std::vector<double> b, bool grad=false, double epsilon = 1e-4) {
    for (int i = 0; i < a.total_count; ++i)
        if (std::fabs((grad? a.values[i]->grad: a.values[i]->data) - b[i]) >= epsilon) {
            std::cerr << "Assertion failed!\n"
                    << "  a = " << a.values[i]->data << "\n"
                    << "  b = " << b[i] << "\n";
            throw std::runtime_error("assert_flatten_tensor_almost_equal");
        }
}


void test_value() {
    // testing operations + * / relu exp zero_grad
    auto x = std::make_shared<Value>(5.0);
    auto y = std::make_shared<Value>(6.0);
    auto z = std::make_shared<Value>(-0.1);

    auto a = x * y;
    auto b = a * z;
    auto c = (b + z)->relu();
    auto e = b + c + a;
    auto final = x / y + z->exp() + e;

    final->backward();
    assert_almost_equal(x->grad, 5.567, 0.001);
    assert_almost_equal(y->grad, 4.361, 0.001);
    assert_almost_equal(z->grad, 30.905, 0.001);
    assert_almost_equal(final->data, 28.738, 0.001);

    final->zero_grad();
    assert_almost_equal(x->grad, 0);
    assert_almost_equal(y->grad, 0);
    assert_almost_equal(z->grad, 0);
}

void test_linear_forward() {
    int in_size = 5, out_size = 5;
    Linear l1(in_size, out_size);
    for (int x = 0; x < l1.out_size; ++x) {
        l1.biases[x]->data = 0.1;
        for (int y = 0; y < l1.in_size; ++y)
            l1.weights[x][y]->data = 0.1 * x + 0.01 * y;
    }

    Tensor input = make_tensor({{0.1, 0.2, 0.3, 0.4, 0.5}});
    Tensor output = l1.forward(input);
    assert_flatten_tensor_almost_equal(output, {0.14, 0.29, 0.44, 0.59, 0.74});
}

void test_softmax() {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});
    Softmax softmax;
    Tensor output = softmax.forward(input);

    assert_flatten_tensor_almost_equal(output, {0.3420, 0.3780, 0.2800});
    output[0][0]->backward();
    assert_flatten_tensor_almost_equal(input, {0.2250, -0.1293, -0.0958}, true);
}

void test_mse() {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});
    Tensor correct = make_tensor({{1, 0, -1}});
    Value_ptr loss = MSELoss(input, correct);
    assert_almost_equal(loss->data, 0.5533);
    loss->backward();
    assert_flatten_tensor_almost_equal(input, {-0.6, 0.1333, 0.6}, true);
}

void test_full_step() {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});

    Linear l1(3, 2);
    Linear l2(2, 2);
    Softmax s;
    SGD optim({l1.params(), l2.params()}, 0.001);

    auto x = l1.forward(input);
    x = l2.forward(x);
    x = s.forward(x);
    Tensor correct = make_tensor({{1, 0}});
    Value_ptr loss = MSELoss(x, correct);
    loss->backward();
    printf("%f %f %f\n", x[0]->data, x[1]->data, loss->data);
    optim.step();

    printf("%f %f", input[0]->data, input[1]->data);
}

void test_tensor() {
    Tensor t({2, 3, 4});
    assert(t.shape == std::vector<int>({2, 3, 4}));
    assert(t.total_count == 24);
    assert(t.values.size() == 24);

    bool caught = false;
    try {
        Tensor t(std::vector<int>({}));
    } catch (...) {
        caught = true;
    }
    assert(caught);

    auto row = t[0];
    assert(row.dim == 1);

    auto elem = t[1][2];
    assert(elem.dim == 2);

    caught = false;
    try {
        auto bad = t[5];
        (void)bad;
    } catch (...) {
        caught = true;
    }
    assert(caught);

    double v1 = 1.5, v2 = 3.7;
    t[0][0][0] = v1;
    t[1][1][3] = v2;

    assert_almost_equal(t.values[0]->data, 1.5);
    assert_almost_equal(t.values[19]->data, 3.7);
}

void test_sgd()
{
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});
    input.values[0]->grad = 0.1;
    input.values[1]->grad = -0.1;
    input.values[2]->grad = 0.5;

    SGD optim({&input}, 0.01);
    optim.step();
    assert_flatten_tensor_almost_equal(input, {0.099, 0.201, -0.105});
}

void test_tensor_operations()
{
    Tensor t1 = make_tensor({{0.1, 0.2, -0.1}, {1, 2, 3}});
    Tensor t2 = make_tensor({{0.3, 0.6, -0.23}, {-1, -2, -3}});
    Tensor t3 = t1 + t2;
    assert_flatten_tensor_almost_equal(t3, {0.4, 0.8, -0.33, 0, 0, 0});
    Tensor t4 = t1 - t2;
    assert_flatten_tensor_almost_equal(t4, {-0.2, -0.4, 0.13, 2, 4, 6});
    Tensor t5 = t1 * t2;
    assert_flatten_tensor_almost_equal(t5, {0.03, 0.12, 0.023, -1, -4, -9});
}

void test_tensor_argmax_2d_last()
{
    Tensor input = make_tensor({{0.1, 0.2, -0.1}, {1, 1, 5}});
    Tensor result = input.argmax(1);
    assert_flatten_tensor_almost_equal(result, {1, 2});
}

int main()
{
    printf("Running tests...\n");
    test_value();
    test_linear_forward();
    test_softmax();
    test_mse();
    test_sgd();
    // tensor
    test_tensor();
    test_tensor_operations();
    test_tensor_argmax_2d_last();
    // test_full_step();
    printf("Tests complete!\n");
}