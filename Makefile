CC = gcc
NVCC = nvcc
CFLAGS = -O3 -ffast-math -Wall
CUDA_FLAGS = -O3 -arch=sm_60

# Targets
all: serial openmp cuda

serial: src/serial/main.c src/serial/gmm.c
	mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o bin/gmm_serial

openmp: src/openmp/main_omp.c src/openmp/gmm_omp.c
	mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp $^ -lm -o bin/gmm_omp

cuda: src/cuda/main_cuda.c src/cuda/gmm_cuda.cu
	mkdir -p bin
	$(NVCC) $(CUDA_FLAGS) $^ -o bin/gmm_cuda -lm

clean:
	rm -rf bin/gmm_serial bin/gmm_omp bin/gmm_cuda
