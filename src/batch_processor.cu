#include <cuda_runtime.h>
#include <nppi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct AppConfig {
    int count = 256;
    int width = 64;
    int height = 64;
    int seed = 42;
    std::string out_dir = "results";
};

void CheckCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

void CheckNpp(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error (" << msg << "): status " << static_cast<int>(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

AppConfig ParseArgs(int argc, char** argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            cfg.count = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            cfg.width = std::max(8, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            cfg.height = std::max(8, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            cfg.seed = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            cfg.out_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: batch_processor [-c count] [-w width] [-h height] [-s seed] [-o out_dir]\n";
            std::exit(EXIT_SUCCESS);
        }
    }
    return cfg;
}

__global__ void ComputeBatchMetricsKernel(const unsigned char* filtered_batch,
                                        float* norm_batch,
                                        float* frame_means,
                                        int count,
                                        int frame_pixels) {
    int frame_idx = blockIdx.y;
    int pixel_in_frame = blockIdx.x * blockDim.x + threadIdx.x;

    if (frame_idx >= count || pixel_in_frame >= frame_pixels) {
        return;
    }

    int global_idx = frame_idx * frame_pixels + pixel_in_frame;
    unsigned char val = filtered_batch[global_idx];
    norm_batch[global_idx] = static_cast<float>(val) / 255.0f;

    if (pixel_in_frame == 0) {
        double sum = 0.0;
        int base = frame_idx * frame_pixels;
        for (int p = 0; p < frame_pixels; ++p) {
            sum += static_cast<double>(filtered_batch[base + p]);
        }
        frame_means[frame_idx] = static_cast<float>(sum / static_cast<double>(frame_pixels));
    }
}

void WriteSamplePgm(const std::string& path, const unsigned char* data, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return;
    }
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(width) * height);
}

}  // namespace

int main(int argc, char** argv) {
    AppConfig cfg = ParseArgs(argc, argv);
    const int frame_pixels = cfg.width * cfg.height;
    const size_t total_pixels = static_cast<size_t>(cfg.count) * static_cast<size_t>(frame_pixels);
    const size_t byte_size = total_pixels * sizeof(unsigned char);
    const size_t float_bytes = total_pixels * sizeof(float);

    std::srand(static_cast<unsigned int>(cfg.seed));
    std::vector<unsigned char> h_input(total_pixels);
    for (size_t i = 0; i < total_pixels; ++i) {
        int base = static_cast<int>(i % 256);
        int noise = (std::rand() % 65) - 32;
        int val = base + noise;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        h_input[i] = static_cast<unsigned char>(val);
    }

    unsigned char* d_input = nullptr;
    unsigned char* d_filtered = nullptr;
    float* d_norm = nullptr;
    float* d_frame_means = nullptr;

    CheckCuda(cudaMalloc(&d_input, byte_size), "cudaMalloc d_input");
    CheckCuda(cudaMalloc(&d_filtered, byte_size), "cudaMalloc d_filtered");
    CheckCuda(cudaMalloc(&d_norm, float_bytes), "cudaMalloc d_norm");
    CheckCuda(cudaMalloc(&d_frame_means, static_cast<size_t>(cfg.count) * sizeof(float)),
              "cudaMalloc d_frame_means");

    cudaStream_t stream;
    CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    NppStreamContext npp_ctx{};
    CheckNpp(nppGetStreamContext(&npp_ctx), "nppGetStreamContext");
    npp_ctx.hStream = stream;

    auto start_time = std::chrono::high_resolution_clock::now();

    CheckCuda(cudaMemcpyAsync(d_input, h_input.data(), byte_size, cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync H2D");

    NppiSize roi{cfg.width, cfg.height};
    NppiSize mask_size{3, 3};
    NppiPoint anchor{1, 1};

    for (int f = 0; f < cfg.count; ++f) {
        size_t offset = static_cast<size_t>(f) * static_cast<size_t>(frame_pixels);
        NppStatus st = nppiFilterBox_8u_C1R_Ctx(
            d_input + offset,
            cfg.width,
            d_filtered + offset,
            cfg.width,
            roi,
            mask_size,
            anchor,
            npp_ctx);
        CheckNpp(st, "nppiFilterBox_8u_C1R_Ctx");
    }

    int threads_x = 256;
    int blocks_x = (frame_pixels + threads_x - 1) / threads_x;
    dim3 grid(blocks_x, cfg.count);
    dim3 block(threads_x, 1);

    ComputeBatchMetricsKernel<<<grid, block, 0, stream>>>(
        d_filtered, d_norm, d_frame_means, cfg.count, frame_pixels);
    CheckCuda(cudaGetLastError(), "ComputeBatchMetricsKernel launch");

    std::vector<unsigned char> h_filtered(total_pixels);
    std::vector<float> h_frame_means(static_cast<size_t>(cfg.count));

    CheckCuda(cudaMemcpyAsync(h_filtered.data(), d_filtered, byte_size, cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync D2H filtered");
    CheckCuda(cudaMemcpyAsync(h_frame_means.data(),
                              d_frame_means,
                              static_cast<size_t>(cfg.count) * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "cudaMemcpyAsync D2H means");

    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    double aggregate_mean = 0.0;
    for (float m : h_frame_means) {
        aggregate_mean += static_cast<double>(m);
    }
    aggregate_mean /= static_cast<double>(cfg.count);

    std::string cmd_mkdir = "mkdir -p " + cfg.out_dir;
    int mk_status = std::system(cmd_mkdir.c_str());
    (void)mk_status;

    std::string log_path = cfg.out_dir + "/execution_log.txt";
    std::ofstream log_file(log_path);
    log_file << "Batch count: " << cfg.count << "\n";
    log_file << "Frame dimensions: " << cfg.width << "x" << cfg.height << "\n";
    log_file << "Total pixels processed: " << total_pixels << "\n";
    log_file << "Aggregate filtered mean intensity: " << aggregate_mean << "\n";
    log_file << "GPU pipeline elapsed (ms): " << elapsed_ms << "\n";

    std::string csv_path = cfg.out_dir + "/frame_means.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "frame_index,mean_intensity\n";
    for (int f = 0; f < cfg.count; ++f) {
        csv_file << f << "," << h_frame_means[f] << "\n";
    }

    WriteSamplePgm(cfg.out_dir + "/sample_frame_0_filtered.pgm", h_filtered.data(), cfg.width, cfg.height);

    std::cout << "Processed " << cfg.count << " frames (" << total_pixels
              << " total pixels) in " << elapsed_ms << " ms | mean intensity: "
              << aggregate_mean << "\n";

    cudaStreamDestroy(stream);
    cudaFree(d_frame_means);
    cudaFree(d_norm);
    cudaFree(d_filtered);
    cudaFree(d_input);
    return 0;
}