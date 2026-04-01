# EC527 Final Project — High-Performance Gaussian Mixture Model (GMM)

**Boston University — EC527: High Performance Programming with Multicore and GPUs**
**Spring 2026 | Professor Herbordt**

**Team**
- Kohki Hatori
- Mimi Paule

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Algorithm Description](#algorithm-description)
3. [Repository Structure](#repository-structure)
4. [Dependencies & Environment](#dependencies--environment)
5. [Building the Project](#building-the-project)
6. [Running the Code](#running-the-code)
7. [Implementations](#implementations)
8. [Validation](#validation)
9. [Optimization Strategies](#optimization-strategies)
10. [Experiments & Results](#experiments--results)
11. [References](#references)

---

## Project Overview

This project implements and optimizes a **Gaussian Mixture Model (GMM)** trained via the **Expectation-Maximization (EM) algorithm**, targeting high-performance execution across three architectures:

| Implementation | Method |
|---|---|
| Serial (baseline) | C, single-threaded |
| Multicore | C + OpenMP |
| GPU | CUDA |

The validation reference is **scikit-learn's** `GaussianMixture`, against which we verify correctness (log-likelihood convergence, component parameters) before measuring performance.

The central HPC challenge in GMM is the **E-step**: for every data point and every mixture component, we compute a full multivariate Gaussian density evaluation — involving matrix-vector products, a quadratic form (Mahalanobis distance), and a log-determinant. This is computationally intensive and highly parallelizable. The **M-step** involves weighted outer products and reductions to update means, covariances, and mixing weights, presenting distinct memory access and synchronization challenges.

---

## Algorithm Description

### Gaussian Mixture Model

A GMM models a distribution as a weighted sum of K multivariate Gaussian components:

```
p(x) = sum_{k=1}^{K} pi_k * N(x | mu_k, Sigma_k)
```

where `pi_k` are mixing weights, `mu_k` are component means, and `Sigma_k` are covariance matrices.

### EM Algorithm

Training iterates between two steps until convergence (change in log-likelihood < epsilon):

**E-step (Expectation):** Compute the responsibility of each component k for each data point n:

```
r_{nk} = pi_k * N(x_n | mu_k, Sigma_k) / sum_{j} pi_j * N(x_n | mu_j, Sigma_j)
```

This requires, for each (n, k) pair:
- Computing the Mahalanobis distance: `(x_n - mu_k)^T Sigma_k^{-1} (x_n - mu_k)`
- Computing `log det(Sigma_k)`
- Applying the log-sum-exp trick for numerical stability

**M-step (Maximization):** Update parameters using the responsibilities:

```
N_k    = sum_n r_{nk}
mu_k   = (1/N_k) * sum_n r_{nk} * x_n
Sigma_k = (1/N_k) * sum_n r_{nk} * (x_n - mu_k)(x_n - mu_k)^T
pi_k   = N_k / N
```

### Complexity

| Step | Complexity |
|---|---|
| E-step (per iteration) | O(N * K * D^2) |
| M-step (per iteration) | O(N * K * D^2) |
| Total | O(I * N * K * D^2) |

where N = number of data points, K = number of components, D = dimensionality, I = number of iterations.

For large N and D (e.g., N=100K, D=64, K=16), this is substantial — and the dominant cost is the E-step matrix operations, which are the primary optimization target.

---

## Repository Structure

```
gmm-hpc/
├── README.md
│
├── src/
│   ├── serial/
│   │   ├── gmm.c              # Serial EM implementation
│   │   ├── gmm.h
│   │   └── main.c             # Driver: load data, run GMM, report timing
│   │
│   ├── openmp/
│   │   ├── gmm_omp.c          # OpenMP-parallelized EM
│   │   ├── gmm_omp.h
│   │   └── main_omp.c
│   │
│   └── cuda/
│       ├── gmm_cuda.cu        # CUDA EM kernels
│       ├── gmm_cuda.cuh
│       ├── kernels/
│       │   ├── estep.cu       # E-step kernel
│       │   ├── mstep.cu       # M-step kernel
│       │   └── logsumexp.cu   # Numerically stable log-sum-exp
│       └── main_cuda.cu
│
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   ├── small/                 # Small test sets (N=1K)
│   ├── medium/                # Medium test sets (N=10K)
│   └── large/                 # Large benchmark sets (N=100K+)
│
├── validation/
│   ├── validate.py            # Compare output vs. scikit-learn reference
│   └── requirements.txt       # Python deps (numpy, scikit-learn)
│
├── results/
│   ├── timing/                # Raw timing CSVs
│   └── plots/                 # Performance plots
│
├── scripts/
│   ├── run_all.sh             # Run all implementations and collect timings
│   └── submit_scc.sh          # SCC job submission script
│
└── Makefile
```

---

## Dependencies & Environment

### BU SCC Cluster Modules

Load the following modules before building:

```bash
module load gcc/12.2.0
module load cuda/12.1
module load python3/3.10.12
```

Add these to your `~/.bashrc` or run at the start of each session.

### Python (validation only)

```bash
pip install --user numpy scikit-learn matplotlib
```

### Hardware Targets

| Resource | Spec |
|---|---|
| Cluster | BU Shared Computing Cluster (SCC) |
| CPU nodes | Intel Xeon (confirm with `lscpu` on your allocation) |
| GPU nodes | NVIDIA (confirm with `nvidia-smi` on your allocation) |

To request an interactive GPU session on SCC:

```bash
qrsh -l gpus=1 -l gpu_c=6.0
```

---

## Building the Project

A single `Makefile` at the project root builds all three targets.

```bash
# Build everything
make all

# Build individual targets
make serial
make openmp
make cuda

# Clean build artifacts
make clean
```

### Compiler flags

| Target | Flags |
|---|---|
| Serial | `gcc -O3 -march=native -ffast-math` |
| OpenMP | `gcc -O3 -march=native -ffast-math -fopenmp` |
| CUDA | `nvcc -O3 -arch=sm_80 --use_fast_math` |

> **Note:** Adjust `-arch=sm_XX` to match the GPU on your SCC allocation. Use `nvidia-smi` to find the GPU model and look up its compute capability.

---

## Running the Code

### Generate synthetic data

```bash
python3 data/generate_data.py --n 10000 --dim 32 --components 8 --out data/medium/
```

### Serial

```bash
./bin/gmm_serial --data data/medium/data.bin --n 10000 --dim 32 --k 8 --iters 100
```

### OpenMP

```bash
export OMP_NUM_THREADS=8
./bin/gmm_omp --data data/medium/data.bin --n 10000 --dim 32 --k 8 --iters 100
```

### CUDA

```bash
./bin/gmm_cuda --data data/medium/data.bin --n 10000 --dim 32 --k 8 --iters 100
```

### Run all and collect timings

```bash
bash scripts/run_all.sh
```

---

## Implementations

### 1. Serial (Baseline)

Straightforward C implementation of EM. Serves as the correctness reference and performance baseline. Includes:
- Cholesky decomposition for stable covariance inversion
- Log-sum-exp trick in the E-step for numerical stability
- Convergence check on log-likelihood delta

### 2. OpenMP (Multicore)

Parallel strategies applied:
- **E-step:** `#pragma omp parallel for` over data points — each thread independently computes responsibilities for its assigned points (no data dependencies across n)
- **M-step:** Parallel reduction for N_k, mu_k, Sigma_k accumulations using `reduction` clause or thread-local accumulators with a final merge
- **Covariance updates:** Parallelized outer product accumulation over N

Thread count controlled via `OMP_NUM_THREADS`. Scaling experiments run from 1 to max available cores.

### 3. CUDA (GPU)

Key kernel designs:

**E-step kernel (`estep.cu`)**
- One thread block per component k, one thread per data point n
- Shared memory used to cache `mu_k` and the pre-computed Cholesky factor of `Sigma_k^{-1}`
- Each thread computes the Mahalanobis distance and log-density for its (n, k) pair
- Log-sum-exp normalization done in a second pass

**M-step kernel (`mstep.cu`)**
- Parallel reduction across N for each component
- Outer product accumulation for covariance uses atomic adds or warp-level reduction

**Memory layout**
- Data matrix stored in column-major order to maximize coalesced access during E-step
- Responsibility matrix `r[N][K]` laid out row-major to allow coalesced writes per data point

---

## Validation

We validate correctness by comparing GMM output against scikit-learn's `GaussianMixture` on the same dataset and initialization.

```bash
# Run serial and save output
./bin/gmm_serial --data data/medium/data.bin ... --out results/serial_out.bin

# Validate against scikit-learn
python3 validation/validate.py \
    --data data/medium/data.bin \
    --our results/serial_out.bin \
    --tol 1e-3
```

Validation checks:
- Final log-likelihood within tolerance of scikit-learn's result
- Component means within L2 tolerance (accounting for label permutation)
- Convergence in a comparable number of iterations

> **Note:** Due to random initialization and floating-point non-determinism, exact parameter matching is not expected. We validate log-likelihood and qualitative cluster recovery.

---

## Optimization Strategies

The following axes are explored across implementations:

| Axis | Serial | OpenMP | CUDA |
|---|---|---|---|
| Algorithmic | Log-sum-exp, Cholesky | — | — |
| Vectorization | `-march=native`, `-ffast-math` | Same | `--use_fast_math` |
| Parallelism | — | Loop parallelism, reductions | Thread blocks, warps |
| Memory layout | — | Cache-friendly access order | Coalesced access, shared mem |
| Blocking | — | Tiling for cache | Shared memory tiling |
| Precision | FP64 vs FP32 comparison | Same | FP32 for GPU throughput |

### Roofline Analysis

We profile each implementation using:
- `perf stat` (CPU, cache misses, FLOP/s estimate)
- `nvprof` / `ncu` (GPU, memory bandwidth, occupancy)

Roofline plots are generated in `results/plots/` to show where each version is bottlenecked (compute-bound vs. memory-bandwidth-bound).

---

## Experiments & Results

*(To be filled in as experiments are completed)*

### Scaling experiments planned

- **Vary N:** 1K, 10K, 50K, 100K data points
- **Vary D:** 2, 16, 32, 64 dimensions
- **Vary K:** 4, 8, 16, 32 components
- **Thread scaling (OpenMP):** 1, 2, 4, 8, 16 threads
- **Batch size (CUDA):** tuning thread block configuration

### Metrics

- Wall-clock time per EM iteration (ms)
- Speedup relative to serial baseline
- GFLOP/s achieved vs. roofline peak
- Memory bandwidth utilization

---

## References

1. Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm.* JRSS-B.
2. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning.* Springer. (Chapter 9: Mixture Models and EM)
3. NVIDIA CUDA Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. scikit-learn GaussianMixture documentation. https://scikit-learn.org/stable/modules/mixture.html
5. Williams, S. et al. (2009). *Roofline: An Insightful Visual Performance Model.* CACM.
