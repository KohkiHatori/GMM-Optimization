# GMM Optimization Analysis: CPU vs GPU Performance & Scaling

## Observation: The "Cross-over Point"
New benchmarking data across a grid of dimensions ($D$) and clusters ($K$) reveals a dynamic performance relationship between the 32-thread OpenMP implementation and the CUDA implementation on a Tesla P100. 

While the OpenMP version is faster for low-complexity configurations (small $D$ and $K$), **the CUDA implementation demonstrates superior scaling** and eventually outperforms the CPU as the computational load increases.

### Performance Inflection Points
*   **Low Complexity ($K < 8$):** OpenMP consistently outperforms CUDA. This is due to the low "arithmetic intensity" relative to the overhead of launching kernels and managing PCIe transfers.
*   **The Cross-over:** As $D$ and $K$ increase, the workload shifts from being memory-latency bound to compute-bound.
    *   At **$D=8$**, the cross-over occurs at **$K=8$**.
    *   At **$D=32$**, the cross-over occurs at **$K=12$** (interpolated between $K=8$ and $K=16$).
*   **High Complexity ($K \ge 32$):** CUDA becomes significantly faster. For example, at **$D=32, K=96$**, the CUDA implementation is **~2.7x faster** than 32-thread OpenMP (9.3s vs 25.5s).

---

## Technical Analysis of the Performance Shift

### 1. Why OpenMP Wins at Low Complexity
*   **Lower Dispatch Latency:** Launching an OpenMP thread pool is significantly faster than initializing a CUDA context and dispatching kernels.
*   **Cache Locality:** For small $K$ and $D$, the entire GMM model (means, covariances, and weights) fits comfortably within the CPU's L2/L3 caches. The CPU can access these parameters with near-zero latency, whereas the GPU must fetch them from HBM2 global memory.
*   **Effective Vectorization:** For $D=32$, the CPU's AVX2/AVX-512 units process the inner loops with extremely high efficiency, essentially negating the GPU's advantage for small workloads.

### 2. Why CUDA Wins at High Complexity
As $O(N \cdot K \cdot D^2)$ grows, the "Compute Intensity" increases. The GPU's massive core count (3,584 cores on a P100) eventually overwhelms the 32-thread CPU once the workload provides enough parallelism to hide the following overheads:
*   **Throughput over Latency:** Once $K$ and $D$ are large enough, the GPU can maintain thousands of active warps, effectively hiding the memory latency of the HBM2.
*   **Compute Dominance:** The $O(K \cdot D^3)$ Cholesky inversions and $O(N \cdot K \cdot D^2)$ outer products become the dominant cost. At $K=96, D=32$, the sheer volume of floating-point operations favors the GPU's higher TFLOPS rating.

---

## Technical Bottlenecks in the Current CUDA Implementation
Despite its scaling advantage, the CUDA version still faces architectural headwinds:
1.  **Memory-Bound Loop:** The dataset is read three times per iteration.
2.  **PCIe Synchronization:** The Cholesky decomposition still happens on the CPU, forcing an expensive `cudaMemcpy` synchronization point in every iteration.
3.  **Atomic Contention:** As $K$ increases, `atomicAdd` operations in shared memory face less contention (since they are spread across more "bins"), which actually helps CUDA scale better at high $K$ than at low $K$.

## Future Work for GPU Acceleration
To push the cross-over point even lower:
1.  **Kernel Fusion:** Combine E-step and M-step to read data once.
2.  **GPU-Side Cholesky:** Use cuSOLVER to keep the entire loop on the GPU.
3.  **Warp-Level Shuffles:** Replace `atomicAdd` with `__shfl_down_sync` for the M-step reductions.
