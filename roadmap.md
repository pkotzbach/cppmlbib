# Roadmap
## CUDA Backend 
- ops from cuda math lib
- CUDA streams and asynchronous memory transfers
- `CudaBuffer` abstraction
- constant memory (`__constant__`) and improved cache utilization

## Architectural Improvements
- add `__restrict__`
- refactor stride calculation logic - isContinous check
- operations (e.g. `argmax`) support across all dimensions
- no-grad mode
- TODOs
- add google benchmark maybe
- common header for tests

*Last Updated: March 22, 2026*
