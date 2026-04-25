#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "gmm_cuda.cuh"

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

#define MAX_D 109 // Limit dictated by 48KB shared memory (109 * 109 * 4 bytes ≈ 47.5KB)
#define HALF_LOG_2PI 0.91893853320467274178f

// Kernel 1: Transpose AoS (N x D) to SoA (D x N) for memory coalescing
__global__ void transpose_aos_to_soa_kernel(const float* aos_data, float* soa_data, int N, int D) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && d < D) {
        soa_data[d * N + n] = aos_data[n * D + d];
    }
}

// Kernel 2: E-Step (Compute Responsibilities and Log-Likelihood)
__global__ void e_step_kernel(const float* soa_data, const float* means, 
                              const float* L_all, const float* log_det_L, 
                              const float* log_weights, float* responsibilities,
                              double* log_likelihood_sum,
                              int N, int D, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float diff[MAX_D];
    float y[MAX_D];
    float max_log_rho = -FLT_MAX;
    float half_d_log_2pi = D * HALF_LOG_2PI;

    for (int k = 0; k < K; k++) {
        // Coalesced read into registers
        for (int d = 0; d < D; d++) {
            diff[d] = soa_data[d * N + n] - means[k * D + d];
        }

        // Solve L_k * y = diff (Forward Sub)
        int L_offset = k * D * D;
        for (int d = 0; d < D; d++) {
            float sum = diff[d];
            for (int j = 0; j < d; j++) {
                sum -= L_all[L_offset + d * D + j] * y[j];
            }
            y[d] = sum / L_all[L_offset + d * D + d];
        }
        
        float maha = 0.0f;
        for (int d = 0; d < D; d++) {
            maha += y[d] * y[d];
        }

        float log_rho = log_weights[k] - half_d_log_2pi - log_det_L[k] - 0.5f * maha;
        responsibilities[n * K + k] = log_rho;
        
        if (log_rho > max_log_rho) {
            max_log_rho = log_rho;
        }
    }

    // Log-sum-exp and Normalization
    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) {
        sum_exp += expf(responsibilities[n * K + k] - max_log_rho);
    }
    float lse = max_log_rho + logf(sum_exp);

    atomicAdd(log_likelihood_sum, (double)lse);

    for (int k = 0; k < K; k++) {
        responsibilities[n * K + k] = expf(responsibilities[n * K + k] - lse);
    }
}

// Kernel 3a: M-Step Pass 1 (Compute Nk and Means)
// Grid: (K, blocks_per_cluster)
// Block: (256, 1, 1)
// Shared Memory: sizeof(double) * (1 + D)
__global__ void m_step_pass1_kernel(const float* soa_data, const float* resp,
                                    double* partial_Nk, double* partial_mu,
                                    int N, int D, int K) {
    int k = blockIdx.x;
    int b = blockIdx.y;
    int num_blocks = gridDim.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ double s_data_p1[];
    double* s_Nk = &s_data_p1[0];
    double* s_mu = &s_data_p1[1];

    if (tid == 0) s_Nk[0] = 0.0;
    for (int i = tid; i < D; i += block_size) s_mu[i] = 0.0;
    __syncthreads();

    double t_Nk = 0.0;

    for (int n = b * block_size + tid; n < N; n += num_blocks * block_size) {
        double r = (double)resp[n * K + k];
        t_Nk += r;

        for (int d = 0; d < D; d++) {
            atomicAdd(&s_mu[d], r * (double)soa_data[d * N + n]);
        }
    }
    
    atomicAdd(&s_Nk[0], t_Nk);
    __syncthreads();

    int offset = k * num_blocks + b;
    if (tid == 0) partial_Nk[offset] = s_Nk[0];
    for (int i = tid; i < D; i += block_size) {
        partial_mu[offset * D + i] = s_mu[i];
    }
}

// Kernel 3b: M-Step Pass 2 (Compute Centered Covariance)
// Grid: (K, blocks_per_cluster)
// Block: (256, 1, 1)
// Shared Memory: sizeof(double) * (D * D)
__global__ void m_step_pass2_kernel(const float* soa_data, const float* resp, const float* means,
                                    double* partial_Sigma,
                                    int N, int D, int K) {
    int k = blockIdx.x;
    int b = blockIdx.y;
    int num_blocks = gridDim.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ double s_Sigma[];

    for (int i = tid; i < D * D; i += block_size) s_Sigma[i] = 0.0;
    __syncthreads();

    for (int n = b * block_size + tid; n < N; n += num_blocks * block_size) {
        double r = (double)resp[n * K + k];
        if (r < 1e-6) continue;

        for (int d1 = 0; d1 < D; d1++) {
            double diff1 = (double)soa_data[d1 * N + n] - (double)means[k * D + d1];
            for (int d2 = 0; d2 <= d1; d2++) {
                double diff2 = (double)soa_data[d2 * N + n] - (double)means[k * D + d2];
                double val = r * diff1 * diff2;
                atomicAdd(&s_Sigma[d1 * D + d2], val);
                if (d1 != d2) {
                    atomicAdd(&s_Sigma[d2 * D + d1], val);
                }
            }
        }
    }
    
    __syncthreads();

    int offset = k * num_blocks + b;
    for (int i = tid; i < D * D; i += block_size) {
        partial_Sigma[offset * D * D + i] = s_Sigma[i];
    }
}

// Host Helper: Compute Cholesky Decomposition of Covariance Matrices
// Computes L_all (lower triangular) and log_det_L on the CPU
void compute_cholesky_host(float* covariances, float* L_all, float* log_det_L, int K, int D) {
    for (int k = 0; k < K; k++) {
        float* L = &L_all[k * D * D];
        float log_det = 0.0f;
        
        for (int i = 0; i < D * D; i++) L[i] = 0.0f;

        for (int i = 0; i < D; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = covariances[k * D * D + i * D + j];
                for (int l = 0; l < j; l++) {
                    sum -= L[i * D + l] * L[j * D + l];
                }
                
                if (i == j) {
                    // Prevent negative or zero due to numerical instability
                    if (sum < 1e-15f) sum = 1e-15f; 
                    L[i * D + j] = sqrtf(sum);
                    log_det += 2.0f * logf(L[i * D + j]);
                } else {
                    L[i * D + j] = sum / L[j * D + j];
                }
            }
        }
        log_det_L[k] = log_det;
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
    // Expected format in validate.py: weights, means, covariances
    fwrite(model->weights, sizeof(float), K, fp);
    fwrite(model->means, sizeof(float), K * D, fp);
    fwrite(model->covariances, sizeof(float), K * D * D, fp);
    fclose(fp);
}

// Main Training Loop
void gmm_train_cuda(float* host_data, int N, int D, int K,
                    int max_iters, float tol,
                    float* init_means, gmm_model_cuda* model) {
    
    if (D > MAX_D) {
        fprintf(stderr, "Error: D=%d exceeds max supported D=%d for shared memory M-step.\n", D, MAX_D);
        exit(EXIT_FAILURE);
    }

    // 1. Host Memory Allocations for EM Loop
    float* h_L_all = (float*)malloc(K * D * D * sizeof(float));
    float* h_log_det_L = (float*)malloc(K * sizeof(float));
    float* h_log_weights = (float*)malloc(K * sizeof(float));
    
    // Initialize Model Parameters
    for (int k = 0; k < K; k++) {
        model->weights[k] = 1.0f / K;
        for (int d = 0; d < D; d++) {
            model->means[k * D + d] = init_means[k * D + d];
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                model->covariances[k * D * D + i * D + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    // Determine blocks for M-step reduction
    int threads_per_block_m = 256;
    int max_blocks_per_cluster = 64; 
    int blocks_per_cluster = (N + threads_per_block_m - 1) / threads_per_block_m;
    if (blocks_per_cluster > max_blocks_per_cluster) blocks_per_cluster = max_blocks_per_cluster;
    
    // Allocate Host arrays for partial reductions
    double* h_partial_Nk = (double*)malloc(K * blocks_per_cluster * sizeof(double));
    double* h_partial_mu = (double*)malloc(K * blocks_per_cluster * D * sizeof(double));
    double* h_partial_Sigma = (double*)malloc(K * blocks_per_cluster * D * D * sizeof(double));

    // 2. Device Memory Allocations
    float *d_data_aos, *d_data_soa, *d_resp;
    float *d_means, *d_L_all, *d_log_det_L, *d_log_weights;
    double *d_partial_Nk, *d_partial_mu, *d_partial_Sigma;
    double *d_ll_sum;

    size_t data_size = N * D * sizeof(float);
    size_t resp_size = N * K * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data_aos, data_size));
    CUDA_CHECK(cudaMalloc(&d_data_soa, data_size));
    CUDA_CHECK(cudaMalloc(&d_resp, resp_size));
    
    CUDA_CHECK(cudaMalloc(&d_means, K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L_all, K * D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_log_det_L, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_log_weights, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ll_sum, sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_partial_Nk, K * blocks_per_cluster * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial_mu, K * blocks_per_cluster * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial_Sigma, K * blocks_per_cluster * D * D * sizeof(double)));

    // 3. Copy Data to Device and Transpose
    CUDA_CHECK(cudaMemcpy(d_data_aos, host_data, data_size, cudaMemcpyHostToDevice));
    
    dim3 block_dim_trans(16, 16);
    dim3 grid_dim_trans((N + block_dim_trans.x - 1) / block_dim_trans.x, (D + block_dim_trans.y - 1) / block_dim_trans.y);
    transpose_aos_to_soa_kernel<<<grid_dim_trans, block_dim_trans>>>(d_data_aos, d_data_soa, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_data_aos)); // Free AoS

    // 4. Execution Setup
    int threads_per_block_e = 256;
    int blocks_per_grid_e = (N + threads_per_block_e - 1) / threads_per_block_e;

    dim3 m_step_grid(K, blocks_per_cluster);
    int shared_mem_p1 = (1 + D) * sizeof(double);
    int shared_mem_p2 = (D * D) * sizeof(double);

    double log_likelihood = -1e18;

    // 5. Training Loop
    for (int iter = 0; iter < max_iters; iter++) {
        double prev_ll = log_likelihood;
        double h_ll_sum = 0.0;
        CUDA_CHECK(cudaMemcpy(d_ll_sum, &h_ll_sum, sizeof(double), cudaMemcpyHostToDevice));

        // --- HOST PREP (Cholesky) ---
        compute_cholesky_host(model->covariances, h_L_all, h_log_det_L, K, D);
        for (int k = 0; k < K; k++) {
            h_log_weights[k] = logf(fmaxf(model->weights[k], 1e-15f));
        }

        CUDA_CHECK(cudaMemcpy(d_means, model->means, K * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_L_all, h_L_all, K * D * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_log_det_L, h_log_det_L, K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_log_weights, h_log_weights, K * sizeof(float), cudaMemcpyHostToDevice));

        // --- E-STEP ---
        e_step_kernel<<<blocks_per_grid_e, threads_per_block_e>>>(
            d_data_soa, d_means, d_L_all, d_log_det_L, d_log_weights, d_resp, d_ll_sum, N, D, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- M-STEP PASS 1 (Nk and Means) ---
        m_step_pass1_kernel<<<m_step_grid, threads_per_block_m, shared_mem_p1>>>(
            d_data_soa, d_resp, d_partial_Nk, d_partial_mu, N, D, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_partial_Nk, d_partial_Nk, K * blocks_per_cluster * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_mu, d_partial_mu, K * blocks_per_cluster * D * sizeof(double), cudaMemcpyDeviceToHost));

        double* h_Nk_all = (double*)malloc(K * sizeof(double));
        for (int k = 0; k < K; k++) {
            double Nk = 0.0;
            for (int b = 0; b < blocks_per_cluster; b++) Nk += h_partial_Nk[k * blocks_per_cluster + b];
            
            h_Nk_all[k] = Nk;
            model->weights[k] = (float)(Nk / N);

            // Aggregate Means
            for (int d = 0; d < D; d++) {
                double mu_sum = 0.0;
                for (int b = 0; b < blocks_per_cluster; b++) {
                    mu_sum += h_partial_mu[(k * blocks_per_cluster + b) * D + d];
                }
                model->means[k * D + d] = (float)(mu_sum / fmax(Nk, 1e-15));
            }
        }

        // Send updated means back to device for Pass 2 (Covariance centering)
        CUDA_CHECK(cudaMemcpy(d_means, model->means, K * D * sizeof(float), cudaMemcpyHostToDevice));

        // --- M-STEP PASS 2 (Covariances) ---
        m_step_pass2_kernel<<<m_step_grid, threads_per_block_m, shared_mem_p2>>>(
            d_data_soa, d_resp, d_means, d_partial_Sigma, N, D, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- HOST FINALIZATION (Covariances) ---
        CUDA_CHECK(cudaMemcpy(h_partial_Sigma, d_partial_Sigma, K * blocks_per_cluster * D * D * sizeof(double), cudaMemcpyDeviceToHost));

        for (int k = 0; k < K; k++) {
            double Nk = h_Nk_all[k];
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    double sig_sum = 0.0;
                    for (int b = 0; b < blocks_per_cluster; b++) {
                        sig_sum += h_partial_Sigma[(k * blocks_per_cluster + b) * D * D + i * D + j];
                    }
                    double centered = sig_sum / fmax(Nk, 1e-15); // Already centered via (X - mu) in pass 2
                    
                    if (i == j) {
                        centered += 1e-4; // Regularization (matches scikit-learn validation setup)
                    }
                    model->covariances[k * D * D + i * D + j] = (float)centered;
                }
            }
        }
        free(h_Nk_all);

        CUDA_CHECK(cudaMemcpy(&h_ll_sum, d_ll_sum, sizeof(double), cudaMemcpyDeviceToHost));
        log_likelihood = h_ll_sum / N;
        printf("  [Iter %3d] Log Likelihood: %lf\n", iter, log_likelihood);

        if (iter > 0 && fabs(log_likelihood - prev_ll) < (double)tol) {
            printf("\nConverged at iteration %d (LL delta < %.1e).\n", iter, tol);
            break;
        }
    }

    // 6. Cleanup
    free(h_L_all);
    free(h_log_det_L);
    free(h_log_weights);
    free(h_partial_Nk);
    free(h_partial_mu);
    free(h_partial_Sigma);

    CUDA_CHECK(cudaFree(d_data_soa));
    CUDA_CHECK(cudaFree(d_resp));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_L_all));
    CUDA_CHECK(cudaFree(d_log_det_L));
    CUDA_CHECK(cudaFree(d_log_weights));
    CUDA_CHECK(cudaFree(d_ll_sum));
    CUDA_CHECK(cudaFree(d_partial_Nk));
    CUDA_CHECK(cudaFree(d_partial_mu));
    CUDA_CHECK(cudaFree(d_partial_Sigma));
}