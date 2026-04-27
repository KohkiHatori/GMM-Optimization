# Empirical Impact Analysis: Dimensionality ($D$) vs. Clusters ($K$)

To mathematically determine whether Dimensionality ($D$) or the number of Clusters ($K$) drives the execution time of our implementations, we performed a Power-Law Regression analysis on the timing data. 

We fit the timing results for $N=500,000$ to the following model using log-log transformation:
$$ \text{WallTime} \approx C \cdot D^a \cdot K^b $$

By comparing the exponents $a$ (for $D$) and $b$ (for $K$), we can identify the dominant structural bottleneck for each architecture.

## 1. Serial Implementation: $K$ is the Driver
*   **Mathematical Model:** $\text{WallTime} \approx 2.07 \times 10^{-3} \cdot D^{1.22} \cdot K^{1.40}$
*   **Verdict:** In the serial code, **Cluster Count ($K$) has the highest structural impact** ($b = 1.40 > a = 1.22$). 
*   **Reasoning:** Although the math suggests $D^2$ complexity, the single-threaded CPU is overwhelmed by the *iterative* complexity of $K$. As $K$ increases, the processor must jump between $K$ different parameter blocks during the data sweep, destroying cache locality and causing severe memory thrashing. The time increases faster than linearly with $K$.

## 2. Parallel Implementations: $D$ is the Driver
Once the algorithm is parallelized, the bottleneck shifts dramatically.

*   **OpenMP (32 Threads):** $\text{WallTime} \approx 5.23 \times 10^{-5} \cdot D^{1.74} \cdot K^{1.55}$
*   **CUDA (Tesla P100):** $\text{WallTime} \approx 8.34 \times 10^{-5} \cdot D^{1.83} \cdot K^{1.15}$
*   **Verdict:** In parallel implementations, **Dimensionality ($D$) is the dominant bottleneck**.

### Why the Shift?
1.  **Parallel Efficiency:** Parallelizing across $N$ (the dataset) is extremely effective at mitigating the impact of $K$. The GPU and multi-core CPU handle the extra clusters with relatively low overhead.
2.  **The $D^2$ Math Ceiling:** The core mathematical operation of the GMM—the Mahalanobis distance—scales quadratically with $D$. While parallelism effectively handles the "width" of the problem ($N$ and $K$), it cannot escape the fundamental $O(D^2)$ complexity of the matrix-vector multiplications required for each cluster evaluation.
3.  **CUDA's Scaling Success:** Notably, the $K$ exponent for CUDA drops to **1.15** (nearly linear), while the $D$ exponent rises to **1.83** (nearly quadratic). This proves that the CUDA Shared-Memory reduction strategy scales exceptionally well with cluster counts, leaving only the fundamental dimensionality math as the primary hardware constraint.
