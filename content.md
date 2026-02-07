## autograd engine
## tensor module
## components
- linear
- softmax
- argmax (2d with last dim)
## optimizers
- sgd
## examples
- iris with ~90% accuracy

## TODO
- classes
- cuda
- use new C++ features
- throw to static_assert
- constexpr
- argmax full
- move tensor operators
- remove total_count
- scalar
    Tensor_ptr div = Tensor::init({1}, std::vector<double>{static_cast<double>(input->shape[0])});
    result = result / div;
- no grad
- broadcasting