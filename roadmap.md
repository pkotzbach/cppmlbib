# Roadmap
## CUDA Backend 
- CUDA streams and asynchronous memory transfers
- `CudaBuffer` abstraction
- constant memory (`__constant__`) and improved cache utilization

## Architectural Improvements
- replace runtime `throw` with `static_assert` where possible
- consistent use of `constexpr`, move semantics, and `__restrict__` pointers for alias-free optimization
- Refactor stride calculation logic - isContinous check
- Operations (e.g. `argmax`) support across all dimensions
- No-grad mode

*Last Updated: March 20, 2026*
