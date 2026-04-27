# Expectation-Maximization (EM) for GMM

The Expectation-Maximization (EM) algorithm is an iterative method to find Maximum Likelihood Estimates (MLE) of parameters in probabilistic models with latent variables (in this case, cluster assignments).

## 1. Parameters and Variables
- **Dataset:** $X = \{x_1, x_2, \dots, x_N\}$, where each $x_i \in \mathbb{R}^D$.
- **Weights ($\pi$):** $\pi_k$ is the prior probability that a point belongs to cluster $k$. $\sum_{k=1}^K \pi_k = 1$.
- **Means ($\mu$):** $\mu_k \in \mathbb{R}^D$ is the center of Gaussian $k$.
- **Covariances ($\Sigma$):** $\Sigma_k \in \mathbb{R}^{D \times D}$ is the covariance matrix of Gaussian $k$.
- **Responsibilities ($\gamma$):** $\gamma_{i,k}$ is the posterior probability (calculated in E-step) that point $i$ belongs to cluster $k$.

---

## 2. The E-Step (Expectation)
In this step, we calculate the "responsibilities" using the current parameter estimates.

### Step A: Multivariate Gaussian Density
For each point $x_i$ and cluster $k$, we define the likelihood $P(x_i | k)$ using the **Multivariate Gaussian Probability Density Function (PDF)**:
$$ P(x_i | k) = \frac{1}{(2\pi)^{D/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x_i - \mu_k)^T \Sigma_k^{-1} (x_i - \mu_k) \right) $$

#### The Mahalanobis Distance
The core of the exponent, $(x_i - \mu_k)^T \Sigma_k^{-1} (x_i - \mu_k)$, is known as the squared **Mahalanobis Distance**. 
Unlike standard Euclidean distance, which treats all directions equally, the Mahalanobis distance uses the inverse covariance matrix ($\Sigma_k^{-1}$) to scale the distance based on the shape of the cluster. If a cluster is stretched out (high variance in a specific direction), it "forgives" distance along that axis, treating points further away as if they were closer to the center.

### Step B: Responsibility Calculation
Using Bayes' Theorem, the posterior probability $\gamma_{i,k}$ is:
$$ \gamma_{i,k} = \frac{\pi_k P(x_i | k)}{\sum_{j=1}^K \pi_j P(x_i | j)} $$

---

## 3. Numerical Implementation & Optimizations

In high-performance GMM implementations, we apply several mathematical tricks to ensure numerical stability and computational efficiency.

### A. Cholesky Decomposition & Log-Determinant Trick
Calculating $P(x_i | k)$ requires inverting the covariance matrix $\Sigma_k$ and finding its determinant $|\Sigma_k|$. Direct matrix inversion is slow and numerically unstable. Instead, we decompose $\Sigma_k$ into a lower-triangular matrix $L$ such that $\Sigma = L L^T$.

**Implementation & $O(D^3)$ Complexity:**
The algorithm builds the lower-triangular matrix $L$ row-by-row. To calculate a single element $L_{i,j}$, it computes a dot product of previously calculated elements in row $i$ and row $j$. This requires a deeply nested triple loop:
1. Iterate over rows $i$ (from $0$ to $D$)
2. Iterate over columns $j$ (from $0$ to $i$)
3. Iterate over the dot-product elements $k$ (from $0$ to $j$)

Because of this 3-level deep nesting, the number of floating-point operations scales cubically, resulting in an algorithmic complexity of $\mathcal{O}(D^3)$. While this becomes a heavy bottleneck at high dimensions (like $D=32$), Cholesky remains the standard because it is roughly twice as fast as general matrix inversion and guarantees mathematical stability.

**The Log-Determinant Optimization:**
In the standard PDF formula, we must multiply by $|\Sigma|^{-1/2}$. In log-space, this becomes $-\frac{1}{2}\ln(|\Sigma|)$. Because the determinant of $\Sigma$ is simply the square of the determinant of $L$ ($|\Sigma| = |L|^2$), the fraction and the exponent cancel each other out beautifully:
$$ -\frac{1}{2}\ln(|L|^2) = -\frac{1}{2} \cdot 2 \cdot \ln(|L|) = -\ln(|L|) $$
Furthermore, since $L$ is a triangular matrix, its determinant is trivially calculated as the product of its diagonal elements. 

**In Code (`src/serial/gmm.c`):**
```c
float log_det = 0.0f;
for (int d = 0; d < D; d++) log_det += logf(L_tmp[d * D + d]); // Sum of log(diagonal)
log_det_L[k] = log_det; 
// ...
float log_rho = log_weights[k] - HALF_D_LOG_2PI - log_det_L[k] - 0.5f * mahalanobis_sq;
```

### B. The Log-Sum-Exp Trick (Normalization)
When calculating $\gamma_{i,k}$, we handle very small probabilities that can lead to **underflow** (numbers becoming $0$ incorrectly). To prevent this, we perform normalization in log-space.

**The Math:**
To calculate $\ln(\sum \exp(v_i))$ safely, we factor out the maximum value $a$:
$$ \ln\sum e^{v_i} = a + \ln\sum e^{v_i - a}, \text{ where } a = \max(v) $$
This ensures that at least one term in the sum is $e^0 = 1$, preventing the sum from becoming zero.

**In Code (`src/serial/gmm.c`):**
```c
// 1. Find max for stability
if (log_rho > max_log_rho) max_log_rho = log_rho;

// 2. Sum shifted exponentials (the denominator)
for (int k = 0; k < K; k++) {
    sum_exp += expf(responsibilities[n * K + k] - max_log_rho);
}
float log_sum_exp_val = max_log_rho + logf(sum_exp);

// 3. Normalize: log(A/B) = log(A) - log(B)
for (int k = 0; k < K; k++) {
    responsibilities[n * K + k] = expf(responsibilities[n * K + k] - log_sum_exp_val);
}
```


---



## 4. The M-Step (Maximization)
In this step, we update the parameters to maximize the expected log-likelihood.

### Step A: Calculate Effective Cluster Size ($N_k$)
$$ N_k = \sum_{i=1}^N \gamma_{i,k} $$

### Step B: Update Weights
$$ \pi_k^{new} = \frac{N_k}{N} $$

### Step C: Update Means
The new mean is the weighted average of all data points:
$$ \mu_k^{new} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{i,k} x_i $$

### Step D: Update Covariances
The new covariance is the weighted outer product of the deviations from the **new** mean:
$$ \Sigma_k^{new} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{i,k} (x_i - \mu_k^{new})(x_i - \mu_k^{new})^T $$

---

## 5. Implementation Strategy: Cache Locality & Accumulators

To handle large datasets ($N > 1,000,000$), the M-step is implemented using a **Two-Pass Accumulator Strategy** to maximize CPU cache performance.

### A. The Role of Accumulators (`acc_`)
Instead of updating the global model parameters directly, we use thread-local or serial "buckets" called accumulators:
- `acc_weights`, `acc_means`, `acc_covs`
These store running sums during the data sweep. This prevents the processor from having to perform frequent, high-latency writes to the main model structures, keeping the "hot" data in the L1/L2 caches.

### B. Two-Pass Strategy for Cache Locality
We split the M-step into two distinct loops:
1. **Pass 1:** Accumulate weights ($N_k$) and means ($\mu_k$).
2. **Pass 2:** Use the *newly finalized* means to accumulate covariances ($\Sigma_k$).

**Why this is faster:**
- **Sequential Access:** By separating the tasks, the code reads the dataset (`data`) and `responsibilities` in a perfectly linear, sequential stride. Modern CPUs have **Hardware Prefetchers** that detect this pattern and proactively load the next chunk of data from DRAM into the cache before the execution core even asks for it.
- **Dependency Resolution:** Calculating $\Sigma_k$ requires $\mu_k^{new}$. By finalizing the means in Pass 1, Pass 2 can use the correct centers immediately, ensuring mathematical correctness while maintaining high memory bandwidth.
- **Branch Prediction:** Separating the mean and covariance logic into simple, tight loops reduces the complexity of the instruction pipeline, allowing the CPU to execute more "Instructions Per Cycle" (IPC).

---

## 6. Convergence
The algorithm repeats the E and M steps until the **Log-Likelihood** converges:
$$ \ln P(X | \pi, \mu, \Sigma) = \sum_{i=1}^N \ln \left( \sum_{k=1}^K \pi_k P(x_i | k) \right) $$
Convergence is typically reached when the increase in log-likelihood between iterations falls below a threshold (e.g., $10^{-6}$).

---

## 7. Computational Complexity & Hardware Bottlenecks

The performance of GMM training is defined by the interaction between the dataset size ($N$), the dimensionality ($D$), and the number of clusters ($K$).

### A. The Per-Iteration Workflow
In every iteration of the EM algorithm, the computational cost is dominated by two distinct phases:
1. **Cluster Preparation (Serial):** Before processing data points, the algorithm must perform $K$ Cholesky decompositions of $D \times D$ covariance matrices.
2. **Data Comparison (Parallel):** The algorithm compares $N$ points against $K$ clusters, involving matrix-vector operations.

### B. Algorithmic Complexity: $\mathcal{O}(I \cdot (K \cdot D^3 + N \cdot K \cdot D^2))$
- **The $K \cdot D^3$ Component:** This represents the Cholesky decomposition. For $D=32$ and $K=96$, this accounts for $\approx 3.1$ million operations per iteration. While numerically smaller than the data loop, these operations are strictly serial for each cluster, creating a synchronization bottleneck.
- **The $N \cdot K \cdot D^2$ Component:** This represents the core E-step and M-step loops. For $N=1,000,000, K=96, D=32$, this scales to $\approx 98$ billion operations per iteration. This phase is highly parallelizable across $N$.

### C. The "Hardware Inflection Point"
The optimal choice of hardware (CPU vs. GPU) depends on these components:
- **CPU Dominance ($D \le 16$ or Low $K$):** At lower complexities, the $K \cdot D^3$ overhead and memory bandwidth are the primary constraints. The CPU's L3 cache and AVX-512 vectorization allow it to outperform GPUs by avoiding PCIe latency and kernel launch overheads.
- **GPU Dominance (High $D$, High $K$):** As $K \cdot D^2$ grows, the raw FLOP throughput of the GPU becomes the deciding factor. The GPU can process the billions of operations in the $N$ loop much faster than the CPU, provided the cluster preparation doesn't become too heavy.

**The CUDA Bottleneck:** A significant challenge for GPU implementations is that standard CUDA kernels are not optimized for many small matrix decompositions. This often results in the CPU handling the $D^3$ math while the GPU handles the $N \cdot K \cdot D^2$ math, requiring frequent data transfers that can limit overall speedup.
