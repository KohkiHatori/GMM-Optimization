# OpenMP Implementation and Optimizations

This document summarizes the parallelization strategies and hardware optimizations used in the OpenMP implementation of the Gaussian Mixture Model (`src/openmp/gmm_omp.c`), specifically targeting high-performance multi-core CPUs.

## 1. Parallelization Strategy: Task Partitioning
The primary parallelization occurs over the dataset size ($N$). Since the calculations for individual data points are mathematically independent in the E-step and the accumulation phase of the M-step, the workload is distributed across multiple CPU threads.

### The "Local-Accumulate then Global-Merge" Pattern
To avoid frequent synchronization overhead, the implementation follows an isolation-first pattern:
1.  **Local Workspace:** Each thread allocates private "accumulator" arrays (`local_Nk`, `local_means`, `local_covs`).
2.  **Autonomous Processing:** Threads process their assigned 1D data slices independently, updating only their local copies.
3.  **Synchronized Merge:** Once a thread finishes its slice, it enters a `#pragma omp critical` block to add its local sums to the global model parameters. This ensures that expensive locking only happens **once per thread** rather than once per data point.

---

## 2. Key OpenMP Pragmas

### A. `#pragma omp parallel reduction(+:log_likelihood)`
This creates a team of threads to process the E-step. The `reduction` clause provides each thread with a private copy of the log-likelihood sum, which is automatically merged (summed) at the end of the parallel region in a thread-safe and efficient manner.

### B. `#pragma omp for schedule(static)`
This distributes the $N$ iterations of the dataset loop. The `static` schedule is chosen because GMM operations per point are computationally uniform. It assigns contiguous chunks of data to each thread, which is highly "cache-friendly" as it maximizes the hit rate of the CPU's L1 and L2 caches.

### C. `#pragma omp simd`
Used in the innermost dimensionality loops (e.g., Mahalanobis distance). This explicitly tells the compiler to utilize **SIMD (Single Instruction, Multiple Data)** hardware units, such as AVX-2 or AVX-512. For $D=32$, this allows the CPU to process up to 16 floating-point operations in a single clock cycle.

### D. `#pragma omp for nowait`
Used in the M-step accumulation loops. By removing the implicit barrier at the end of the loop, threads that finish their data slice early can immediately move to the `critical` section to merge their results, reducing idle "wait" time and improving overall throughput.

---

## 3. Cache and Memory Optimizations

### Two-Pass M-Step Strategy
The M-step is split into two distinct passes to maximize **Cache Locality**:
- **Pass 1:** Accumulates Weights and Means.
- **Pass 2:** Accumulates Covariances.

**Benefits:**
- **Sequential Stride:** By separating these tasks, the CPU reads the massive $N \times D$ data matrix in a linear fashion. This triggers the CPU's **Hardware Prefetcher**, which proactively loads data from DRAM into the cache.
- **Register Efficiency:** Keeping the loops simple allows the compiler to fit more variables into CPU registers, reducing the number of slow "spills" to memory.

### SIMD Alignment ($D=32$)
The choice of $D=32$ as a benchmark metric is strategic. $32$ floats equal $128$ bytes, which aligns perfectly with modern cache line sizes (64 or 128 bytes) and vector register widths (256-bit or 512-bit). This ensures that the CPU never wastes a vector cycle on "partial" data loads.

---

## 4. Serial vs. Parallel Boundaries
While the $N$-loop is fully parallelized, certain components remain **serial** for stability and correctness:
1.  **Cholesky Decomposition:** Executed serially for each of the $K$ clusters before the E-step. While $D^3$ is expensive, the serial overhead is outweighed by the $N \cdot K$ parallel gains.
2.  **Global Aggregation:** The final division of accumulators by the total weight ($N_k$) happens on a single thread to ensure the model parameters are updated atomically before the next iteration.
