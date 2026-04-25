# GMM Optimization Analysis: D vs. K Grid Scaling

## Objective
To deeply understand the hardware limitations and scaling characteristics of the Gaussian Mixture Model (GMM) implementations, we conducted a 2D grid search over the parameter space. Rather than just comparing execution times for a single configuration, this grid search maps the entire performance landscape across varying **Dimensions ($D$)** and **Cluster Counts ($K$)**.

## Methodology
The benchmark fixes the dataset size to $N=500,000$ points and sweeps across a $5 \times 5$ grid:
*   **Dimensions ($D$):** 4, 8, 16, 32, 64
*   **Clusters ($K$):** 2, 4, 8, 16, 32

We profile three implementations: Serial (baseline), OpenMP (24 threads representing peak CPU), and CUDA (Tesla P100). The results are visualized as a **3D Surface Plot** ($X=K, Y=D, Z=\text{Wall Time}$).

## Architectural Bottlenecks Analyzed

### 1. Scaling Dimensions ($D$): The Shift to Compute-Bound
The number of dimensions exponentially increases the arithmetic intensity of the Expectation-Maximization (EM) algorithm.
*   **The Math:** During the E-step, computing the multivariate Gaussian PDF requires a Cholesky decomposition and matrix inversion of the covariance matrix, which scales at $\mathcal{O}(D^3)$. Computing the Mahalanobis distance and updating the covariance matrix in the M-step scales at $\mathcal{O}(D^2)$.
*   **The Hardware Reality:**
    *   **Low $D$ (Memory Bound):** At $D \le 16$, the math is trivial. The execution time is entirely dominated by memory bandwidth (streaming the $N$ points). The 24-thread CPU excels here because its massive L3 cache and hardware prefetchers easily handle small, contiguous memory loads, heavily outperforming the GPU's PCIe transfer latency and global memory accesses.
    *   **High $D$ (Compute Bound):** As $D$ approaches 32 and 64, the $\mathcal{O}(D^3)$ matrix inversions overwhelm the CPU's vector units. This represents the **Inflection Point** where the GPU's massive floating-point (FLOP) throughput finally eclipses the CPU's cache advantage.

### 2. Scaling Clusters ($K$): Shared Memory and Atomic Contention
The number of clusters dictates the size of the working memory required per block and the degree of thread serialization.
*   **The Math:** The overall EM algorithm scales linearly, $\mathcal{O}(K)$, meaning more clusters roughly linearly increase the number of distances to calculate.
*   **The Hardware Reality:**
    *   **CPU (Thread-Local Independence):** The OpenMP implementation allocates thread-local arrays (`local_cov`) for accumulation. As $K$ increases, the CPU simply uses slightly more of its L1/L2 cache. It remains entirely lock-free until the final reduction.
    *   **GPU (Atomic Contention):** In the CUDA M-step reduction, the cluster means and covariances are accumulated in Shared Memory. As $K$ increases, more threads attempt to use `atomicAdd` on the exact same shared memory addresses (specifically, the covariance bins). This creates massive hardware serialization and bank conflicts. Higher $K$ acts as a friction multiplier on the GPU's M-step.

## Conclusion and The 3D Inflection Point
By visualizing the data as overlapping 3D surfaces, the "Inflection Point" becomes mathematically visible:
1.  **CPU Domain:** The CPU surface remains flat and extremely fast in the "Low $D$, Low $K$" quadrant.
2.  **GPU Domain:** The CPU surface violently spikes upwards in the "High $D$" region due to $\mathcal{O}(D^3)$ complexity, while the GPU surface rises much more gradually.
3.  **Trade-offs:** If a workload involves massive $N$ but small $D$ (e.g., Image Quantization, $D=3$), heavily threaded CPUs will often defeat standard GPU kernels due to caching. If a workload involves high $D$ (e.g., Audio MFCCs, $D=39$), the GPU becomes strictly necessary to handle the arithmetic load.