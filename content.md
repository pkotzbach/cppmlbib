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

## ai usage
- tests
- benchmark.cpp

## TODO
### cuda
- cuda launchers?
- cuda streams
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#async-execution-memory-transfers
- page locked memory https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#explicit-memory-management
- cache memory https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#caches
- cuda_check?
- tests refactor (CUDA_TEST?)
- expect_cuda = false update - should i even have cuda for all tests? (e.g argmax)

### c++
- use new C++ features
- throw to static_assert
- constexpr
- move
- Use __restrict__
- warnings - changing int to size_t increases memory

### rest
- {} consistency andTyping also_consistency
- double or float? Global memory is accessed via 32-byte memory transactions. https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access
- what about stride in softmax for example? new result should have same stride or no?
- https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions
- TODOs
- argmax full
- move tensor operators
- remove total_count
- scalar
    Tensor_ptr div = Tensor::init({1}, std::vector<double>{static_cast<double>(input->shape[0])});
    result = result / div;
- no grad
- faster at() and strides
- consts Each SM also has a separate constant cache, which is used to cache values in global memory that have been declared to be constant over the life of a kernel
- values_vec to continous_values?
- tests should have common header with this CUDA stuff