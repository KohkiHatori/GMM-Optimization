#ifndef GMM_H
#define GMM_H

/* 
 * GMM Model 
 * Using a 1D flat array for maximum cache efficiency and layout matching with GPU.
 * Dimensions:
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
void gmm_free(gmm_model* model);

// Train the model
// data is expected to be a flat N x D array of single precision floats
void gmm_train(float* data, int N, int D, int K, int max_iters, float tol, float* init_means, gmm_model* model);
void gmm_save(gmm_model* model, int K, int D, const char* filepath);

#endif // GMM_H
