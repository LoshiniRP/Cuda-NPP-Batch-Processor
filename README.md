# Cuda-NPP-Batch-Processor

Processes a batch of 2D frames/patches (256 frames of `64x64`, or 1,048,576 total pixels by default) on the GPU using NVIDIA Performance Primitives (`nppiFilterBox_8u_C1R_Ctx`) over a `cudaStream_t` paired with a custom CUDA `__global__` normalization and per-frame mean kernel.

## Repository Layout
- `src/batch_processor.cu`: CUDA/NPP batch processing implementation and CLI argument parser
- `Makefile`: Build and clean targets
- `run.sh`: Automated build and execution script
- `INSTALL`: Environment setup instructions
- `results/`: Execution log, per-frame CSV summary, and sample filtered frame

## Usage
```bash
chmod +x run.sh
./run.sh