#include "value.hpp"
#include "linear.hpp"
#include <iostream>
#include <cmath>

inline void assert_almost_equal(double a, double b, double epsilon = 1e-4) {
    if (std::fabs(a - b) >= epsilon) {
        std::cerr << "Assertion failed!\n"
                  << "  a = " << a << "\n"
                  << "  b = " << b << "\n";
        throw std::runtime_error("assert_almost_equal");
    }
}

inline Value_ptr val(double x) {
    return std::make_shared<Value>(x);
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
    Linear l1(5, 5);
    for (int x = 0; x < l1.out_size; ++x) {
        l1.biases[x]->data = 0.1;
        for (int y = 0; y < l1.in_size; ++y)
            l1.weights[x][y]->data = 0.1 * x + 0.01 * y;
    }

    std::vector<Value_ptr> input = { val(0.1), val(0.2), val(0.3), val(0.4), val(0.5) };
    auto output = l1.forward(input);
    double correct_output[5] = {0.14, 0.29, 0.44, 0.59, 0.74};

    for (int x = 0; x < 5; ++x)
        assert_almost_equal(correct_output[x], output[x]->data);

}

void test_softmax() {
    const int size = 3;
    std::vector<Value_ptr> input = { val(0.1), val(0.2), val(-0.1)};
    Softmax softmax(size);
    std::vector<Value_ptr> output = softmax.forward(input);
    double correct_output[size] = {0.3420, 0.3780, 0.2800};

    for (int x = 0; x < size; ++x)
        assert_almost_equal(correct_output[x], output[x]->data);

    output[0]->backward();
    double correct_grad[size] = {0.2250, -0.1293, -0.0958};

    for (int x = 0; x < size; ++x)
        assert_almost_equal(correct_grad[x], input[x]->grad);
}

void test_full_forward() {
    std::vector<Value_ptr> input = { val(0.1), val(0.2), val(-0.1)};
    Linear l1(3, 2);
    Linear l2(2, 2);
    Softmax s(2);

    auto x = l1.forward(input);
    x = l2.forward(x);
    x = s.forward(x);

    printf("%f %f", x[0]->data, x[1]->data);
}

int main()
{
    test_value();
    test_linear_forward();
    test_softmax();
    test_full_forward();
}