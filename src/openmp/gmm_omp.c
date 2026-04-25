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

// Invert DxD lower triangular matrix L into L_inv
static void invert_lower_triangular(const float* L, float* L_inv, int D) {
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
    float* Sigma_inv_all    = (float*)malloc((size_t)K * D * D * sizeof(float));
    float* log_det_L        = (float*)malloc(K * sizeof(float));
    float* log_weights      = (float*)malloc(K * sizeof(float));
    float* L_tmp            = (float*)malloc(D * D * sizeof(float));
    float* L_inv_tmp        = (float*)malloc(D * D * sizeof(float));

    int max_threads = omp_get_max_threads();
    float* thread_diffs = (float*)malloc(max_threads * D * sizeof(float));

    const float HALF_D_LOG_2PI = 0.5f * (float)D * logf(2.0f * PI);
    double log_likelihood = -1e18;

    // EM LOOP
    for (int iter = 0; iter < max_iters; iter++) {
        double prev_ll = log_likelihood;
        log_likelihood = 0.0;

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

            float ld = 0.0f;
            for (int d = 0; d < D; d++) ld += logf(L_tmp[d*D+d]);
            log_det_L[k] = ld;                       // log|Σ_k|^½ = Σ log L_kk

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
        #pragma omp parallel reduction(+:log_likelihood)
        {
            int tid = omp_get_thread_num();
            float* diff = &thread_diffs[tid * D];

            #pragma omp for schedule(static)
            for (int n = 0; n < N; n++) {
                const float* xn = &data[n * D];
                float max_log_rho = -FLT_MAX;

                // Compute unnormalised log-density for each component k
                for (int k = 0; k < K; k++) {
                    for (int d = 0; d < D; d++)
                        diff[d] = xn[d] - model->means[k*D+d];

                    // Mahalanobis² = diff^T * Sigma_inv * diff
                    float maha = 0.0f;
                    for (int d1 = 0; d1 < D; d1++) {
                        float s = 0.0f;
                        #pragma omp simd
                        for (int d2 = 0; d2 < D; d2++) {
                            s += diff[d2] * Sigma_inv_all[k*D*D + d1*D + d2];
                        }
                        maha += diff[d1] * s;
                    }

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

                log_likelihood += (double)lse;               // reduced across threads

                // Convert log-responsibilities to normalised probabilities
                for (int k = 0; k < K; k++)
                    responsibilities[n*K+k] = expf(responsibilities[n*K+k] - lse);
            }
        } // end E-step parallel region

        log_likelihood /= N;   // report per-point average

        // M-STEP Optimized: Inverted loops for better cache locality and N-parallelism
        double* acc_weights = (double*)calloc(K, sizeof(double));
        double* acc_means = (double*)calloc(K * D, sizeof(double));
        double* acc_covs = (double*)calloc(K * D * D, sizeof(double));

        // Pass 1: Accumulate Nk and Means (Parallel over N)
        #pragma omp parallel
        {
            double* local_Nk = (double*)calloc(K, sizeof(double));
            double* local_means = (double*)calloc(K * D, sizeof(double));

            #pragma omp for nowait
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    double resp = (double)responsibilities[n * K + k];
                    local_Nk[k] += resp;
                    for (int d = 0; d < D; d++) {
                        local_means[k * D + d] += resp * (double)data[n * D + d];
                    }
                }
            }

            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    acc_weights[k] += local_Nk[k];
                    for (int d = 0; d < D; d++) {
                        acc_means[k * D + d] += local_means[k * D + d];
                    }
                }
            }
            free(local_Nk);
            free(local_means);
        }

        // Finalize means and calculate inverse Nk for covariance pass
        double* inv_Nk_all = (double*)malloc(K * sizeof(double));
        for (int k = 0; k < K; k++) {
            double Nk = acc_weights[k];
            inv_Nk_all[k] = (Nk > 1e-10) ? 1.0 / Nk : 0.0;
            model->weights[k] = (float)(Nk / N); // Update global weights
            for (int d = 0; d < D; d++) {
                model->means[k * D + d] = (float)(acc_means[k * D + d] * inv_Nk_all[k]);
            }
        }

        // Pass 2: Accumulate Covariances (Parallel over N)
        #pragma omp parallel
        {
            double* local_cov = (double*)calloc(K * D * D, sizeof(double));
            #pragma omp for nowait
            for (int n = 0; n < N; n++) {
                const float* x_n = &data[n * D];
                for (int k = 0; k < K; k++) {
                    double resp = (double)responsibilities[n * K + k];
                    if (resp < 1e-6) continue; // Skip negligible contributions
                    const float* mu_k = &model->means[k * D];
                    for (int d1 = 0; d1 < D; d1++) {
                        double diff1 = (double)x_n[d1] - (double)mu_k[d1];
                        for (int d2 = 0; d2 <= d1; d2++) { // Lower triangle only
                            local_cov[k * D * D + d1 * D + d2] += resp * diff1 * ((double)x_n[d2] - (double)mu_k[d2]);
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    for (int d1 = 0; d1 < D; d1++) {
                        for (int d2 = 0; d2 <= d1; d2++) {
                            acc_covs[k * D * D + d1 * D + d2] += local_cov[k * D * D + d1 * D + d2];
                        }
                    }
                }
            }
            free(local_cov);
        }

        // Finalize covariances (scale and symmetrize)
        for (int k = 0; k < K; k++) {
            double inv_Nk = inv_Nk_all[k];
            for (int d1 = 0; d1 < D; d1++) {
                for (int d2 = 0; d2 <= d1; d2++) {
                    double val = acc_covs[k * D * D + d1 * D + d2] * inv_Nk;
                    model->covariances[k * D * D + d1 * D + d2] = (float)val;
                    model->covariances[k * D * D + d2 * D + d1] = (float)val; // Symmetrize
                }
            }
        }
        free(inv_Nk_all);
        free(acc_weights);
        free(acc_means);
        free(acc_covs);

        // Output progress
        printf("  [Iter %3d] Log Likelihood: %lf\n", iter, log_likelihood);

        if (iter > 0 && fabs(log_likelihood - prev_ll) < (double)tol) {
            printf("\nConverged at iteration %d (LL delta < %.1e).\n", iter, tol);
            break;
        }
    }

    free(responsibilities);
    free(Sigma_inv_all);
    free(log_det_L);
    free(log_weights);
    free(L_tmp);
    free(L_inv_tmp);
    free(thread_diffs);
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
