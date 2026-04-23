#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gmm_cuda.h"

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel 1: Transpose AoS (N x D) to SoA (D x N) for memory coalescing
__global__ void transpose_aos_to_soa_kernel(const float* aos_data, float* soa_data, int N, int D) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && d < D) {
        // AoS index: n * D + d
        // SoA index: d * N + n
        soa_data[d * N + n] = aos_data[n * D + d];
    }
}

// Kernel 2: E-Step (Compute Responsibilities)
// Maps nicely to the GPU: 1 thread per data point N
__global__ void e_step_kernel(const float* soa_data, const float* means, 
                              const float* L_all, const float* log_det_L, 
                              const float* log_weights, float* responsibilities, 
                              int N, int D, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Use Registers instead of malloc'd buffers
    float diff[MAX_D]; // MAX_D must be known at compile time or use shared memory
    float y[MAX_D];
    float max_log_rho = -3.40282e38f; // FLT_MAX

    for (int k = 0; k < K; k++) {
        // COALESCED READ: soa_data[d*N + n]
        for (int d = 0; d < D; d++)
            diff[d] = soa_data[d*N + n] - means[k*D + d];

        // Solve L_k · y = diff (Forward Sub)
        // Port your forward_sub logic here
        
        float maha = 0.0f;
        for (int d = 0; d < D; d++) maha += y[d] * y[d];

        float log_rho = log_weights[k] - HALF_D_LOG_2PI - log_det_L[k] - 0.5f * maha;
        responsibilities[n*K + k] = log_rho;
        if (log_rho > max_log_rho) max_log_rho = log_rho;
    }

    // Log-sum-exp and Normalization
    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++)
        sum_exp += expf(responsibilities[n*K + k] - max_log_rho);
    float lse = max_log_rho + logf(sum_exp);

    for (int k = 0; k < K; k++)
        responsibilities[n*K + k] = expf(responsibilities[n*K + k] - lse);
}

// Kernel 3: M-Step (Update Parameters)
__global__ void m_step_update_kernel(const float* soa_data, const float* resp,
                                     float* new_means, float* Nk_acc, 
                                     int N, int D, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    for (int k = 0; k < K; k++) {
        float r = resp[n*K + k];
        atomicAdd(&Nk_acc[k], r); // Accumulate effective count

        for (int d = 0; d < D; d++) {
            // Update means using atomicAdd
            atomicAdd(&new_means[k*D + d], r * soa_data[d*N + n]);
        }
    }
}

// Host Functions
gmm_model_cuda* gmm_init_cuda(int K, int D) {
    gmm_model_cuda* model = (gmm_model_cuda*)malloc(sizeof(gmm_model_cuda));
    model->means = (float*)malloc(K * D * sizeof(float));
    model->covariances = (float*)malloc(K * D * D * sizeof(float));
    model->weights = (float*)malloc(K * sizeof(float));
    return model;
}

void gmm_free_cuda(gmm_model_cuda* model) {
    if (model) {
        free(model->means);
        free(model->covariances);
        free(model->weights);
        free(model);
    }
}

void gmm_save_cuda(gmm_model_cuda* model, int K, int D, const char* filepath) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening %s for writing\n", filepath);
        return;
    }
    fwrite(model->means, sizeof(float), K * D, fp);
    fwrite(model->covariances, sizeof(float), K * D * D, fp);
    fwrite(model->weights, sizeof(float), K, fp);
    fclose(fp);
}

// Main Training Loop
void gmm_train_cuda(float* host_data, int N, int D, int K,
                    int max_iters, float tol,
                    float* init_means, gmm_model_cuda* model) {
    
    // 1. Allocate Device Memory
    float *d_data_aos, *d_data_soa, *d_resp;
    float *d_means, *d_covs, *d_weights;

    size_t data_size = N * D * sizeof(float);
    size_t resp_size = N * K * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data_aos, data_size));
    CUDA_CHECK(cudaMalloc(&d_data_soa, data_size));
    CUDA_CHECK(cudaMalloc(&d_resp, resp_size));
    
    CUDA_CHECK(cudaMalloc(&d_means, K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covs, K * D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));

    // 2. Copy Data to Device (AoS format)
    CUDA_CHECK(cudaMemcpy(d_data_aos, host_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, init_means, K * D * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize weights evenly
    float* host_weights = (float*)malloc(K * sizeof(float));
    for (int k = 0; k < K; k++) host_weights[k] = 1.0f / K;
    CUDA_CHECK(cudaMemcpy(d_weights, host_weights, K * sizeof(float), cudaMemcpyHostToDevice));
    free(host_weights);

    // 3. Transpose Data (AoS -> SoA)
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (D + block_dim.y - 1) / block_dim.y);
    transpose_aos_to_soa_kernel<<<grid_dim, block_dim>>>(d_data_aos, d_data_soa, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // We no longer need AoS on the device
    CUDA_CHECK(cudaFree(d_data_aos));

    // 4. Execution Setup
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 5. Training Loop
    for (int iter = 0; iter < max_iters; iter++) {
        
        // E-Step
        e_step_kernel<<<blocks_per_grid, threads_per_block>>>(
            d_data_soa, d_means, d_covs, d_weights, d_resp, N, D, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Zero out parameters before M-Step (Implementation dependent)
        CUDA_CHECK(cudaMemset(d_means, 0, K * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_weights, 0, K * sizeof(float)));
        
        // M-Step
        m_step_kernel<<<blocks_per_grid, threads_per_block>>>(
            d_data_soa, d_resp, d_means, d_covs, d_weights, N, D, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // TODO: M-Step Normalization & Log-Likelihood convergence check
    }

    // 6. Copy Results Back to Host
    CUDA_CHECK(cudaMemcpy(model->means, d_means, K * D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(model->covariances, d_covs, K * D * D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(model->weights, d_weights, K * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_data_soa));
    CUDA_CHECK(cudaFree(d_resp));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_covs));
    CUDA_CHECK(cudaFree(d_weights));
}