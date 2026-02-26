## autograd engine
## tensor module
- broadcasting (only at add)
## components
- linear
- softmax
- argmax (2d with last dim)
## optimizers
- sgd
## examples
- iris with ~90% accuracy

## TODO
- cuda
- unify ops
- what about stride in softmax for example? new result should have same stride or no?
- https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions
- TODOs
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
- more broadcasting
- faster at()
- move
- consts Each SM also has a separate constant cache, which is used to cache values in global memory that have been declared to be constant over the life of a kernel
- warnings - changing int to size_t increases memory