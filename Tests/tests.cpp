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

void test_value() {
    // testing operations + * / relu exp zero_grad
    Value x(5), y(6), z(-0.1);

    Value a = x * y;
    Value b = a * z;
    Value c = b + z;
    c = c.relu();
    Value d = b + c;
    Value e = d + a;
    Value f = x / y;
    Value g = z.exp();
    Value h = f + g;
    Value final = e + h;

    final.backward();

    // printf("%f\n", x.grad);
    // printf("%f\n", y.grad);
    // printf("%f\n", z.grad);
    // printf("%f\n", final.data);
    assert_almost_equal(x.grad, 5.567, 0.001);
    assert_almost_equal(y.grad, 4.361, 0.001);
    assert_almost_equal(z.grad, 30.905, 0.001);
    assert_almost_equal(final.data, 28.738, 0.001);

    final.zero_grad();
    assert_almost_equal(x.grad, 0);
    assert_almost_equal(y.grad, 0);
    assert_almost_equal(z.grad, 0);
}

void test_linear_forward() {
    Linear l1(5, 5);
    for (int x = 0; x < l1.out_size; ++x) {
        l1.biases[x] = 0.1;
        for (int y = 0; y < l1.in_size; ++y)
            l1.weights[x][y].data = 0.1 * x + 0.01 * y;
    }

    Value input[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    Value* output = l1.forward(input);
    double correct_output[5] = {0.14, 0.29, 0.44, 0.59, 0.74};

    for (int x = 0; x < 5; ++x)
        assert_almost_equal(correct_output[x], output[x].data);

    delete[] output;
}

void test_softmax() {
    const int size = 3;
    Value input[size] = {0.1, 0.2, -0.1};
    Softmax softmax(size);
    Value* output = softmax.forward(input);
    double correct_output[size] = {0.3420, 0.3780, 0.2800};

    for (int x = 0; x < size; ++x)
        assert_almost_equal(correct_output[x], output[x].data);

    output[0].backward();
    double correct_grad[size] = {0.2250, -0.1293, -0.0958};

    for (int x = 0; x < size; ++x)
        assert_almost_equal(correct_grad[x], output[x].grad);

    delete[] output;
}

int main()
{
    test_value();
    // test_linear_forward();
    // test_softmax();
}