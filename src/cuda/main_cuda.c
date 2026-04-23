#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "gmm_cuda.h"

double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
        quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);

    struct timespec temp;
    temp.tv_sec = time_stop.tv_sec - time_start.tv_sec;
    temp.tv_nsec = time_stop.tv_nsec - time_start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    meas = (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
    j *= 2;
  }
  return quasi_random;
}

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

int main(int argc, char** argv) {
    char data_path[256] = "";
    char out_path[256] = "";
    char init_path[256] = "";
    int N = 10000;
    int D = 32;
    int K = 8;
    int max_iters = 100;
    float tol = 1e-5f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            strcpy(data_path, argv[++i]);
        } else if (strcmp(argv[i], "--init") == 0 && i + 1 < argc) {
            strcpy(init_path, argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            D = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fit_tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strcpy(out_path, argv[++i]);
        }
    }

    if (strlen(data_path) == 0) {
        printf("Usage: %s --data <path> [--n N] [--dim D] [--k K] [--iters I]\n", argv[0]);
        return 1;
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("CUDA Device: %s\n", prop.name);
    }

    printf("Loading flat binary data from %s...\n", data_path);
    FILE* fp = fopen(data_path, "rb");
    if (!fp) {
        printf("Failed to open %s\n", data_path);
        return 1;
    }

    float* data = (float*)malloc(N * D * sizeof(float));
    size_t actually_read = fread(data, sizeof(float), N * D, fp);
    fclose(fp);

    if (actually_read != (size_t)(N * D)) {
        printf("Warning: Expected %d floats but read %zu.\n", N * D, actually_read);
    }

    // CPU WAKEUP
    double qr = wakeup_delay();

    gmm_model_cuda* model = gmm_init_cuda(K, D);

    struct timespec time_start, time_stop;

    printf("Starting CUDA GMM Training (N=%d, D=%d, K=%d, Iters=%d)...\n\n", N, D, K, max_iters);
    
    // Parse Initialization Means
    float* init_means = NULL;
    if (strlen(init_path) > 0) {
        printf("Loading random initial means from %s...\n", init_path);
        init_means = (float*)malloc(K * D * sizeof(float));
        FILE* fp_init = fopen(init_path, "rb");
        if (fp_init) {
            fread(init_means, sizeof(float), K * D, fp_init);
            fclose(fp_init);
        }
    }

    // Start Timer
    clock_gettime(CLOCK_REALTIME, &time_start);

    // Start CUDA EM Algorithm
    gmm_train_cuda(data, N, D, K, max_iters, tol, init_means, model);

    // Stop Timer
    clock_gettime(CLOCK_REALTIME, &time_stop);
    double elapsed = interval(time_start, time_stop);

    printf("\nExecution Context: qr=%f\n", qr);
    printf("Total Wall Time : %.4f seconds\n", elapsed);

    if (strlen(out_path) > 0) {
        gmm_save_cuda(model, K, D, out_path);
    }

    gmm_free_cuda(model);
    free(data);
    if (init_means) free(init_means);

    return 0;
}