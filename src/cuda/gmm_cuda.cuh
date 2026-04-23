#ifndef GMM_CUDA_H
#define GMM_CUDA_H

/*
 * GMM Model (Host-side representation for CUDA implementation)
 * These point to host (CPU) memory. The GPU training function 
 * will handle the cudaMalloc and cudaMemcpy operations internally.
 *
 * means:       K x D
 * covariances: K x D x D
 * weights:     K
 */
typedef struct {
    float* means;
    float* covariances;
    float* weights;
} gmm_model_cuda;

#ifdef __cplusplus
extern "C" {
#endif

// Initialization and destruction (Allocates host memory)
gmm_model_cuda* gmm_init_cuda(int K, int D);
void            gmm_free_cuda(gmm_model_cuda* model);

/*
 * Train with CUDA parallelism.
 * * NOTE ON MEMORY LAYOUT: 
 * 'data' is passed in as your standard AoS (N x D) flat array from the host.
 * For optimal GPU memory coalescing, this function should allocate device 
 * memory and transpose 'data' into an SoA (D x N) format before launching 
 * the E-step and M-step kernels.
 */
void gmm_train_cuda(float* data, int N, int D, int K,
                    int max_iters, float tol,
                    float* init_means, gmm_model_cuda* model);

// Save the trained model parameters to a binary file
void gmm_save_cuda(gmm_model_cuda* model, int K, int D, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // GMM_CUDA_H