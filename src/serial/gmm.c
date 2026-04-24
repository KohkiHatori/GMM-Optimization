#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "gmm.h"

#define PI 3.14159265358979323846f

gmm_model* gmm_init(int K, int D) {
    gmm_model* model = (gmm_model*)malloc(sizeof(gmm_model));
    model->means = (float*)calloc(K * D, sizeof(float));
    model->covariances = (float*)calloc(K * D * D, sizeof(float));
    model->weights = (float*)calloc(K, sizeof(float));
    return model;
}

void gmm_free(gmm_model* model) {
    if (model) {
        if (model->means) free(model->means);
        if (model->covariances) free(model->covariances);
        if (model->weights) free(model->weights);
        free(model);
    }
}

// Cholesky decomposition of a DxD symmetric positive-definite matrix A
// Result is stored in L (lower triangular)
// Returns 1 if successful, 0 if matrix is not positive-definite.
int cholesky(float* A, float* L, int D) {
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i * D + k] * L[j * D + k];
            }
            if (i == j) {
                float val = A[i * D + i] - sum;
                if (val <= 0.0f) return 0; // Not positive definite
                L[i * D + j] = sqrtf(val);
            } else {
                L[i * D + j] = (1.0f / L[j * D + j]) * (A[i * D + j] - sum);
            }
        }
    }
    return 1;
}

// Invert DxD lower triangular matrix L into L_inv
void invert_lower_triangular(const float* L, float* L_inv, int D) {
    memset(L_inv, 0, D * D * sizeof(float));
    for (int i = 0; i < D; i++) {
        L_inv[i*D+i] = 1.0f / L[i*D+i];
        for (int j = 0; j < i; j++) {
            float s = 0.0f;
            for (int k = j; k < i; k++) {
                s += L[i*D+k] * L_inv[k*D+j];
            }
            L_inv[i*D+j] = -s / L[i*D+i];
        }
    }
}

void gmm_train(float* data, int N, int D, int K, int max_iters, float tol, float* init_means, gmm_model* model) {
    if (init_means != NULL) {
        for (int k = 0; k < K; k++) {
            model->weights[k] = 1.0f / K;
            for (int d = 0; d < D; d++) {
                model->means[k * D + d] = init_means[k * D + d];
                for (int d2 = 0; d2 < D; d2++) {
                    if (d == d2) {
                        model->covariances[k * D * D + d * D + d2] = 1.0f; // Identity init
                    } else {
                        model->covariances[k * D * D + d * D + d2] = 0.0f;
                    }
                }
            }
        }
    } else {
        // 1. Random Initialization
        srand(42);
        // Shuffle an array of indices to pick K unique points
        int* indices = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) indices[i] = i;
        for (int i = 0; i < K; i++) {
            int swap_idx = i + rand() % (N - i);
            int temp = indices[i];
            indices[i] = indices[swap_idx];
            indices[swap_idx] = temp;
        }
        
        // Set means to K random points, weights to 1/K, cov to identity
        for (int k = 0; k < K; k++) {
            model->weights[k] = 1.0f / K;
            for (int d = 0; d < D; d++) {
                model->means[k * D + d] = data[indices[k] * D + d];
                for (int d2 = 0; d2 < D; d2++) {
                    if (d == d2) {
                        model->covariances[k * D * D + d * D + d2] = 1.0f; // Identity init
                    } else {
                        model->covariances[k * D * D + d * D + d2] = 0.0f;
                    }
                }
            }
        }
        free(indices);
    }

    float* responsibilities = (float*)malloc(N * K * sizeof(float));
    float* Sigma_inv_all = (float*)malloc(K * D * D * sizeof(float));
    float* diff = (float*)malloc(D * sizeof(float));
    float* log_weights = (float*)malloc(K * sizeof(float));
    float* log_det_L = (float*)malloc(K * sizeof(float));
    float* L_tmp = (float*)malloc(D * D * sizeof(float));
    float* L_inv_tmp = (float*)malloc(D * D * sizeof(float));
    
    float log_likelihood = -1e9;
    const float HALF_D_LOG_2PI = 0.5f * (float)D * logf(2.0f * PI);

    // EM LOOP
    for (int iter = 0; iter < max_iters; iter++) {
        float prev_log_likelihood = log_likelihood;
        log_likelihood = 0.0f;
        
        // E-STEP Preparations: invert covariances and compute determinants
        for (int k = 0; k < K; k++) {
            log_weights[k] = logf(fmaxf(model->weights[k], 1e-15f));
            
            // Add tiny regularization to diagonal for numeric stability
            for (int d = 0; d < D; d++) {
                model->covariances[k * D * D + d * D + d] += 1e-4f;
            }
            
            // Re-initialize L matrix for this K
            for (int d = 0; d < D * D; d++) L_tmp[d] = 0.0f;
            
            if(!cholesky(&model->covariances[k * D * D], L_tmp, D)) {
                // If singularity occurs, add strong Tikhonov regularization block
                for (int d = 0; d < D; d++) {
                    model->covariances[k * D * D + d * D + d] += 1.0f; 
                }
                cholesky(&model->covariances[k * D * D], L_tmp, D); // Recompute
            }
            
            float log_det = 0.0f;
            for (int d = 0; d < D; d++) log_det += logf(L_tmp[d * D + d]);
            log_det_L[k] = log_det; 

            // Precompute inverse covariance matrix: Sigma_inv = L_inv^T * L_inv
            invert_lower_triangular(L_tmp, L_inv_tmp, D);
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    float s = 0.0f;
                    int max_ij = (i > j) ? i : j;
                    for (int m = max_ij; m < D; m++) {
                        s += L_inv_tmp[m*D+i] * L_inv_tmp[m*D+j];
                    }
                    Sigma_inv_all[k*D*D + i*D + j] = s;
                }
            }
        }

        // E-STEP Core: evaluate density and normalize
        for (int n = 0; n < N; n++) {
            float max_log_rho = -FLT_MAX;
            
            // For each point, calculate the probability belonging to each cluster K
            for (int k = 0; k < K; k++) {
                for (int d = 0; d < D; d++) {
                    diff[d] = data[n * D + d] - model->means[k * D + d];
                }
                
                // Mahalanobis² = diff^T * Sigma_inv * diff
                float mahalanobis_sq = 0.0f;
                for (int d1 = 0; d1 < D; d1++) {
                    float s = 0.0f;
                    for (int d2 = 0; d2 < D; d2++) {
                        s += diff[d2] * Sigma_inv_all[k*D*D + d1*D + d2];
                    }
                    mahalanobis_sq += diff[d1] * s;
                }
                
                float log_rho = log_weights[k] - HALF_D_LOG_2PI - log_det_L[k] - 0.5f * mahalanobis_sq;
                responsibilities[n * K + k] = log_rho;
                
                if (log_rho > max_log_rho) max_log_rho = log_rho;
            }
            
            // Log-sum-exp trick to safely normalize probabilities
            float sum_exp = 0.0f;
            for (int k = 0; k < K; k++) {
                sum_exp += expf(responsibilities[n * K + k] - max_log_rho);
            }
            float log_sum_exp_val = max_log_rho + logf(sum_exp);
            
            log_likelihood += log_sum_exp_val;
            
            // Finalize responsibility matrix by exponentiating and subtracting log normalizer
            for (int k = 0; k < K; k++) {
                responsibilities[n * K + k] = expf(responsibilities[n * K + k] - log_sum_exp_val);
            }
        }
        
        log_likelihood /= N; // Average log likelihood

        // M-STEP Optimized: Cache locality and symmetry
        memset(model->weights, 0, K * sizeof(float));
        memset(model->means, 0, K * D * sizeof(float));
        memset(model->covariances, 0, K * D * D * sizeof(float));

        // Pass 1: Accumulate Nk and Means (Outer loop over N)
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                float resp = responsibilities[n * K + k];
                model->weights[k] += resp;
                for (int d = 0; d < D; d++) {
                    model->means[k * D + d] += resp * data[n * D + d];
                }
            }
        }

        // Finalize means and calculate inverse Nk for covariance pass
        float* inv_Nk_all = (float*)malloc(K * sizeof(float));
        for (int k = 0; k < K; k++) {
            float Nk = model->weights[k];
            inv_Nk_all[k] = (Nk > 1e-10f) ? 1.0f / Nk : 0.0f;
            model->weights[k] = Nk / N; // Update global weights
            for (int d = 0; d < D; d++) {
                model->means[k * D + d] *= inv_Nk_all[k];
            }
        }

        // Pass 2: Accumulate Covariances (Outer loop over N)
        for (int n = 0; n < N; n++) {
            const float* x_n = &data[n * D];
            for (int k = 0; k < K; k++) {
                float resp = responsibilities[n * K + k];
                if (resp < 1e-6f) continue; // Skip negligible contributions
                const float* mu_k = &model->means[k * D];
                for (int d1 = 0; d1 < D; d1++) {
                    float diff1 = x_n[d1] - mu_k[d1];
                    for (int d2 = 0; d2 <= d1; d2++) { // Lower triangle only
                        model->covariances[k * D * D + d1 * D + d2] += resp * diff1 * (x_n[d2] - mu_k[d2]);
                    }
                }
            }
        }

        // Finalize covariances (scale and symmetrize)
        for (int k = 0; k < K; k++) {
            float inv_Nk = inv_Nk_all[k];
            for (int d1 = 0; d1 < D; d1++) {
                for (int d2 = 0; d2 <= d1; d2++) {
                    float val = model->covariances[k * D * D + d1 * D + d2] * inv_Nk;
                    model->covariances[k * D * D + d1 * D + d2] = val;
                    model->covariances[k * D * D + d2 * D + d1] = val; // Symmetrize
                }
            }
        }
        free(inv_Nk_all);
        
        // Output progress
        printf("  [Iter %3d] Log Likelihood: %f\n", iter, log_likelihood);
        
        if (iter > 0 && fabsf(log_likelihood - prev_log_likelihood) < tol) {
            printf("\nConverged at iteration %d (LL Delta < %.1e).\n", iter, tol);
            break;
        }
    }

    free(responsibilities);
    free(Sigma_inv_all);
    free(diff);
    free(log_weights);
    free(log_det_L);
    free(L_tmp);
    free(L_inv_tmp);
}

void gmm_save(gmm_model* model, int K, int D, const char* filepath) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        printf("Failed to open %s for saving model.\n", filepath);
        return;
    }
    
    fwrite(model->weights, sizeof(float), K, fp);
    fwrite(model->means, sizeof(float), K * D, fp);
    fwrite(model->covariances, sizeof(float), K * D * D, fp);
    
    fclose(fp);
    printf("Saved output model parameters to %s\n", filepath);
}

