# Cuda-NPP-Batch-Processor

A GPU-accelerated batch image processing pipeline built with **CUDA** and **NVIDIA Performance Primitives (NPP)**. By default, it processes **256 frames** of **`64x64`** patches (**1,048,576 total pixels**) asynchronously over a dedicated `cudaStream_t`, combining NPP box filtering (`nppiFilterBox_8u_C1R_Ctx`) with a custom CUDA `__global__` normalization and per-frame mean intensity kernel.

## Overview & Pipeline Architecture

The application is designed to keep batch processing cohesive on the device and minimize host-device synchronization overhead:

1. **Host Batch Preparation**: Allocates and populates host buffers for the batch of 2D single-channel (`8u_C1`) frames according to the configured frame count and dimensions.
2. **Asynchronous H2D Transfer**: Copies input frame data to GPU memory (`cudaMemcpyAsync`) on a dedicated `cudaStream_t`.
3. **NPP Spatial Filtering**: Applies a `3x3` box filter (`nppiFilterBox_8u_C1R_Ctx`) with anchor `(1, 1)` over the region of interest (`NppiSize`) using an explicitly configured `NppStreamContext` (`npp_ctx.hStream = stream`).
4. **Custom CUDA Kernel (`__global__`)**: Runs a custom device kernel to normalize pixel values and compute per-frame mean intensities across the batch.
5. **D2H Readback & Output Generation**: Transfers computed results back to the host, logs execution metrics, writes per-frame mean intensities to CSV, and exports a sample filtered frame in PGM format.

## Repository Layout

* `src/batch_processor.cu`: CUDA/NPP batch processing pipeline, custom `__global__` kernel, stream context setup, and CLI argument parser.
* `Makefile`: Build and clean targets (`nvcc` compilation and linking against `cudart`, `nppif`, and `nppc`).
* `run.sh`: Automated build and execution script that runs the batch processor and populates `results/`.
* `INSTALL`: Environment and toolchain setup notes.
* `results/`: Sample execution outputs (`execution_log.txt`, `frame_means.csv`, and `sample_frame_0_filtered.pgm`).

## Building and Running

### Quick Start via `run.sh`

```bash
chmod +x run.sh
./run.sh
```

### Manual Build with Makefile
```bash
make
./batch_processor
```

### To clean compiled artifacts:
```bash
make clean
```

### Direct nvcc Compilation
```bash
mkdir -p results
nvcc -std=c++17 -O2 -I/usr/local/cuda/include src/batch_processor.cu -o batch_processor -L/usr/local/cuda/lib64 -lcudart -lnppif -lnppc
./batch_processor
```


## Command-Line Options & Configuration
**src/batch_processor.cu** parses command-line arguments to control batch execution parameters (such as frame count, frame width/height, and execution options—inspect **src/batch_processor.cu** or run with **--help** for details). Default execution runs a 256-frame batch at **64x64** resolution (1,048,576 total pixels).

## Notes on CUDA / NPP Stream Context Compatibility
When building against newer CUDA/NPP toolkits (such as CUDA 13.0), **NppStreamContext** is initialized directly and bound to the active CUDA stream via **npp_ctx.hStream = stream;** before launching stream-aware NPP routines, avoiding reliance on **nppGetStreamContext.** 

## Sample Execution Results (NVIDIA Tesla T4)
Verified on Google Colab using an NVIDIA Tesla T4 GPU and nvcc 13.0:
**Metric**                              **Value**
Batch frame count                          256
Frame dimensions                          64x64
Total pixels processed                  1,048,576
GPU pipeline elapsed time               25.5945ms
Aggregate filtered mean intensity        127.019

## Generated Artifacts in results/
**results/execution_log.txt:** Summary of batch configuration, total pixel count, aggregate filtered mean intensity, and GPU pipeline elapsed time.
**results/frame_means.csv:** Per-frame mean intensity values (frame_index,mean_intensity) for frames 0 through 255.
**results/sample_frame_0_filtered.pgm:** Filtered output image for sample frame 0 in PGM format.

## Peer Review Checklist
- Public repository URL pointing to the main branch of Cuda-NPP-Batch-Processor
- Complete source code (src/batch_processor.cu), build/run files (Makefile, run.sh, INSTALL), and README.md
- Command-line argument support in src/batch_processor.cu
- Execution proof artifacts (results/execution_log.txt, results/frame_means.csv, results/sample_frame_0_filtered.pgm)
