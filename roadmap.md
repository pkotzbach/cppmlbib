# Roadmap
## CUDA Backend 
- ops from cuda math lib
- CUDA streams and asynchronous memory transfers
- latency hiding
- `CudaBuffer` abstraction
- constant memory (`__constant__`) and improved cache utilization
- GEMM upgrades: warp tiling

## Architectural Improvements
- refactor stride calculation logic - isContinous check
- operations (e.g. `argmax`) support across all dimensions
- no-grad mode
- TODOs
- add google benchmark maybe