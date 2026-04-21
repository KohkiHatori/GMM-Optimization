#ifndef GMM_OMP_H
#define GMM_OMP_H

/*
 * GMM Model (shared layout with serial version)
 * Flat 1-D arrays for cache efficiency and easy GPU porting later.
 *   means:       K x D
 *   covariances: K x D x D
 *   weights:     K
 */
typedef struct {
    float* means;
    float* covariances;
    float* weights;
} gmm_model;

// Initialization and destruction
gmm_model* gmm_init(int K, int D);
void       gmm_free(gmm_model* model);

// Train with OpenMP parallelism.
// data is a flat N x D array of single-precision floats.
void gmm_train(float* data, int N, int D, int K,
               int max_iters, float tol,
               float* init_means, gmm_model* model);

void gmm_save(gmm_model* model, int K, int D, const char* filepath);

#endif // GMM_OMP_H
