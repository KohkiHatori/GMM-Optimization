#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "gmm_omp.h"

#define PI 3.14159265358979323846f

gmm_model* gmm_init(int K, int D) {
    gmm_model* model   = (gmm_model*)malloc(sizeof(gmm_model));
    model->means       = (float*)calloc(K * D,     sizeof(float));
    model->covariances = (float*)calloc(K * D * D, sizeof(float));
    model->weights     = (float*)calloc(K,          sizeof(float));
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
static int cholesky(const float* A, float* L, int D) {
    memset(L, 0, D * D * sizeof(float));
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            float s = 0.0f;
            for (int k = 0; k < j; k++) s += L[i*D+k] * L[j*D+k];
            if (i == j) {
                float v = A[i*D+i] - s;
                if (v <= 0.0f) return 0;   // not positive-definite
                L[i*D+j] = sqrtf(v);
            } else {
                L[i*D+j] = (A[i*D+j] - s) / L[j*D+j];
            }
        }
    }
    return 1;
}

// Forward substitution to solve L * y = x
// L is DxD lower triangular, x is Dx1 input, y is Dx1 output
static void forward_sub(const float* L, const float* x, float* y, int D) {
    for (int i = 0; i < D; i++) {
        float s = 0.0f;
        for (int j = 0; j < i; j++) s += L[i*D+j] * y[j];
        y[i] = (x[i] - s) / L[i*D+i];
    }
}

// USING OPENMP!!!! :D
void gmm_train(float* data, int N, int D, int K,
               int max_iters, float tol,
               float* init_means, gmm_model* model)
{
    /* ---- Initialisation (identical to serial baseline) ---- */
    if (init_means != NULL) {
        for (int k = 0; k < K; k++) {
            model->weights[k] = 1.0f / K;
            for (int d = 0; d < D; d++) {
                model->means[k*D+d] = init_means[k*D+d];
                for (int d2 = 0; d2 < D; d2++)
                    model->covariances[k*D*D + d*D + d2] = (d == d2) ? 1.0f : 0.0f;
            }
        }
    } else {
        srand(42);
        int* indices = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) indices[i] = i;
        for (int i = 0; i < K; i++) {
            int j = i + rand() % (N - i);
            int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
        }
        for (int k = 0; k < K; k++) {
            model->weights[k] = 1.0f / K;
            for (int d = 0; d < D; d++) {
                model->means[k*D+d] = data[indices[k]*D+d];
                for (int d2 = 0; d2 < D; d2++)
                    model->covariances[k*D*D + d*D + d2] = (d == d2) ? 1.0f : 0.0f;
            }
        }
        free(indices);
    }

    /* ---- Shared workspace ---- */
    float* responsibilities = (float*)malloc((size_t)N * K * sizeof(float));
    float* L_all            = (float*)malloc((size_t)K * D * D * sizeof(float));
    float* log_det_L        = (float*)malloc(K * sizeof(float));
    float* log_weights      = (float*)malloc(K * sizeof(float));
    float* L_tmp            = (float*)malloc(D * D * sizeof(float));

    const float HALF_D_LOG_2PI = 0.5f * (float)D * logf(2.0f * PI);
    float log_likelihood = -1e9f;

    // EM LOOP
    for (int iter = 0; iter < max_iters; iter++) {
        float prev_ll = log_likelihood;
        log_likelihood = 0.0f;

        // E-STEP Preparations: invert covariances and compute determinants
        for (int k = 0; k < K; k++) {
            log_weights[k] = logf(fmaxf(model->weights[k], 1e-15f));

            // Add tiny regularization to diagonal for numeric stability
            for (int d = 0; d < D; d++)
                model->covariances[k*D*D + d*D + d] += 1e-4f;

            memset(L_tmp, 0, D * D * sizeof(float));
            if (!cholesky(&model->covariances[k*D*D], L_tmp, D)) {
                // If singularity occurs, add strong Tikhonov regularization block
                for (int d = 0; d < D; d++)
                    model->covariances[k*D*D + d*D + d] += 1.0f;
                cholesky(&model->covariances[k*D*D], L_tmp, D); // Recompute
            }

            memcpy(&L_all[k*D*D], L_tmp, D * D * sizeof(float));

            float ld = 0.0f;
            for (int d = 0; d < D; d++) ld += logf(L_tmp[d*D+d]);
            log_det_L[k] = ld;                       // log|Σ_k|^½ = Σ log L_kk
        }

        // E-STEP Core: evaluate density and normalize
        #pragma omp parallel reduction(+:log_likelihood)
        {
            // Per-thread scratch buffers (allocated once, reused every iteration)
            float* diff = (float*)malloc(D * sizeof(float));
            float* y    = (float*)malloc(D * sizeof(float));

            #pragma omp for schedule(static)
            for (int n = 0; n < N; n++) {
                const float* xn = &data[n * D];
                float max_log_rho = -FLT_MAX;

                // Compute unnormalised log-density for each component k
                for (int k = 0; k < K; k++) {
                    for (int d = 0; d < D; d++)
                        diff[d] = xn[d] - model->means[k*D+d];

                    // Solve L_k · y = diff  →  y = L_k^{-1}(x-μ_k)
                    forward_sub(&L_all[k*D*D], diff, y, D);

                    // Mahalanobis² = ||y||²  (because ||L^{-1}(x-μ)||² = (x-μ)ᵀΣ^{-1}(x-μ))
                    float maha = 0.0f;
                    for (int d = 0; d < D; d++) maha += y[d] * y[d];

                    float log_rho = log_weights[k] - HALF_D_LOG_2PI
                                  - log_det_L[k] - 0.5f * maha;
                    responsibilities[n*K+k] = log_rho;
                    if (log_rho > max_log_rho) max_log_rho = log_rho;
                }

                // Log-sum-exp trick: stable normalisation
                float sum_exp = 0.0f;
                for (int k = 0; k < K; k++)
                    sum_exp += expf(responsibilities[n*K+k] - max_log_rho);
                float lse = max_log_rho + logf(sum_exp);

                log_likelihood += lse;               // reduced across threads

                // Convert log-responsibilities to normalised probabilities
                for (int k = 0; k < K; k++)
                    responsibilities[n*K+k] = expf(responsibilities[n*K+k] - lse);
            }

            free(diff);
            free(y);
        } // end E-step parallel region

        log_likelihood /= N;   // report per-point average

        // M-STEP
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < K; k++) {
            // Effective count for component k
            float Nk = 0.0f;
            for (int n = 0; n < N; n++) Nk += responsibilities[n*K+k];

            float inv_Nk = (Nk > 1e-10f) ? 1.0f / Nk : 0.0f;

            // Update weights
            model->weights[k] = Nk / N;

            // Update means
            float* mu_k = &model->means[k*D];
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int n = 0; n < N; n++)
                    acc += responsibilities[n*K+k] * data[n*D+d];
                mu_k[d] = acc * inv_Nk;
            }

            // Update covariances
            float* cov_k = &model->covariances[k*D*D];
            for (int d1 = 0; d1 < D; d1++) {
                for (int d2 = 0; d2 < D; d2++) {
                    float acc = 0.0f;
                    for (int n = 0; n < N; n++) {
                        float v1 = data[n*D+d1] - mu_k[d1];
                        float v2 = data[n*D+d2] - mu_k[d2];
                        acc += responsibilities[n*K+k] * v1 * v2;
                    }
                    cov_k[d1*D+d2] = acc * inv_Nk;
                }
            }
        } // end M-step parallel region

        // Output progress
        printf("  [Iter %3d] Log Likelihood: %f\n", iter, log_likelihood);

        if (iter > 0 && fabsf(log_likelihood - prev_ll) < tol) {
            printf("\nConverged at iteration %d (LL delta < %.1e).\n", iter, tol);
            break;
        }
    }

    free(responsibilities);
    free(L_all);
    free(log_det_L);
    free(log_weights);
    free(L_tmp);
}

//I/O

void gmm_save(gmm_model* model, int K, int D, const char* filepath) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) { printf("Failed to open %s for saving model.\n", filepath); return; }
    fwrite(model->weights,     sizeof(float), K,     fp);
    fwrite(model->means,       sizeof(float), K*D,   fp);
    fwrite(model->covariances, sizeof(float), K*D*D, fp);
    fclose(fp);
    printf("Saved output model parameters to %s\n", filepath);
}
