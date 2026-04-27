# CUDA Implementation and Optimizations

This document summarizes the architectural decisions and optimizations used in the CUDA implementation of the Gaussian Mixture Model (`src/cuda/gmm_cuda.cu`), specifically targeting NVIDIA GPUs like the Tesla P100.

## 1. Data Layout: SoA vs. AoS
The CPU implementation uses an Array of Structures (AoS) format (e.g., `[x, y, z], [x, y, z]`), which is optimal for CPU cache lines. However, GPUs require memory to be accessed in contiguous chunks by groups of 32 threads (Warps). 

Before the EM loop begins, the CUDA implementation launches a transpose kernel to convert the dataset into a **Structure of Arrays (SoA)** format (e.g., `[x, x], [y, y], [z, z]`). This enables **Coalesced Memory Access**, allowing the GPU to fetch data from the HBM2 memory with maximum efficiency.

---

## 2. Grid and Block Structures

### A. E-Step: 1D Grid
In the E-step, the operations per data point are entirely independent. 
- **Mapping:** 1 Thread = 1 Data Point.
- **Grid Layout:** 1-Dimensional. For $N = 1,000,000$, the grid consists of $\lceil N / 256 \rceil \approx 3,907$ blocks.
- **Reduction:** To compute the global sum for the Log-Likelihood, threads use `atomicAdd` on a single global variable, bypassing the need for complex block-level synchronization.

### B. M-Step: 2D Grid
The M-step requires aggregating data across the entire dataset for $K$ distinct clusters. To prevent massive atomic contention on global memory, a 2D grid is utilized:
- **X-Dimension (Clusters):** Each column maps to a specific cluster $k$.
- **Y-Dimension (Data Slices):** The $N$ dataset is divided into slices (capped at 64 slices per cluster).
- **Why 2D?** This allows the kernel to easily partition work. A thread block knows exactly which cluster it is updating and which portion of the 1M points it is responsible for sweeping.

---

## 3. Shared Memory and Reduction
In the OpenMP M-step, each thread allocates large private arrays to avoid race conditions. GPUs cannot do this due to severe memory limits per thread.

Instead, the CUDA M-step uses **Block-Level Reduction with Shared Memory**:
1. A block of 256 threads is allocated a single `__shared__` array (e.g., `s_Sigma`) in the ultra-fast L1 scratchpad memory.
2. The 256 threads process their data slice and use `atomicAdd` to write to the `s_Sigma` array. Because this is shared memory, atomic collisions are resolved exceptionally fast.
3. Once all threads finish (`__syncthreads()`), the block copies its final `s_Sigma` partial sum back to global memory.
4. The CPU handles the final aggregation of these (up to 64) partial sums.

---

## 4. Hardware Sizing: Why 256 Threads per Block?
While NVIDIA GPUs support up to 1024 threads per block, the implementation specifically hardcodes `threads_per_block_e = 256`. This is optimized for the **Tesla P100** (Pascal architecture) for three primary reasons:

1. **Register Pressure:** The P100 Streaming Multiprocessor (SM) has 65,536 registers. At $D=32$, the E-step requires over 64 registers per thread just for local array allocations (`diff`, `y`). If 1024 threads were used, the register demand ($1024 \times \sim80$) would exceed the SM limit, causing catastrophic "register spilling" to slow global memory. 256 threads keep the register count safe.
2. **Occupancy & Latency Hiding:** A P100 SM can hold 2048 active threads. Using 256-thread blocks allows the SM to perfectly fit 8 blocks simultaneously. This provides a massive pool of warps; when one block is waiting for memory, the GPU instantly switches to a ready block, effectively hiding memory latency.
3. **Symmetry:** 256 is a clean divisor of the 2048 limit. Pushing thread counts higher risks dropping the number of active blocks on the SM drastically if register limits are slightly breached.

---

## 5. CPU-GPU Hybrid Math
The CUDA implementation does **not** perform the Cholesky Decomposition on the GPU. 
Cholesky requires deep, sequential, triangular loops that scale cubically ($O(D^3)$). GPUs excel at massive parallel math but stall severely on serial dependencies. 

To optimize this, the algorithm uses a hybrid approach:
1. The GPU finishes the M-Step.
2. Covariances are copied to the CPU.
3. The CPU computes the $K$ Cholesky decompositions using fast serial operations.
4. The resulting $L$ matrices are copied back to the GPU for the E-Step.
