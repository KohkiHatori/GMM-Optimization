# Arithmetic Intensity Derivation for GMM

To understand the hardware scaling limits of our Gaussian Mixture Model (GMM) implementation, we must analyze its **Arithmetic Intensity (AI)**. Arithmetic Intensity is defined as the ratio of computational work (Floating Point Operations, or FLOPs) to memory traffic (Bytes transferred from main memory).

$$ \text{Arithmetic Intensity (AI)} = \frac{\text{Total FLOPs}}{\text{Total Memory Traffic (Bytes)}} $$

The following derivation focuses on the **E-Step core loop**, which is the dominant computational bottleneck of the algorithm.

---

## 1. Total FLOPs (The Work)
For each data point $x_n$, the algorithm must evaluate the Multivariate Gaussian PDF against every cluster $k$. The dominant operation here is the **Mahalanobis distance**: $(x_n - \mu_k)^T \Sigma_k^{-1} (x_n - \mu_k)$.

Let's break down the operations for a single data point evaluated against a single cluster:
1. **Subtract Mean ($x_n - \mu_k$):** $D$ operations.
2. **Matrix-Vector Product ($v = \Sigma_k^{-1} \cdot \text{diff}$):** Each of the $D$ elements in the resulting vector requires a dot product of length $D$.
   - $D$ multiplications + $(D-1)$ additions $\approx 2D$ operations per element.
   - Total for $D$ elements: **$2D^2$ operations**.
3. **Final Dot Product ($\text{diff}^T \cdot v$):** $D$ multiplications + $(D-1)$ additions $\approx$ **$2D$ operations**.
4. **Log-Sum-Exp Normalization:** $\approx$ **$K$ operations**.

For a single data point compared against all $K$ clusters, the highest order term is the matrix-vector product.
**Total FLOPs per data point** $\approx K \cdot (2D^2 + 3D) \approx \mathbf{2 K D^2}$

---

## 2. Total Memory Traffic (The Bytes)
Next, we count the "Compulsory" bytes that must move from DRAM (Main Memory) to the processor registers. 

*Assumption: The model parameters ($\mu, \Sigma^{-1}$) are relatively small and accessed repeatedly. Thus, we assume they reside in the L2/L3 Cache (CPU) or Shared Memory (GPU), making their amortized transfer cost per data point negligible.*

1. **Read Data Point ($x_n$):** $D$ floats $\times$ 4 bytes/float = **$4D$ bytes**.
2. **Write Responsibilities ($\gamma_{n,k}$):** $K$ floats $\times$ 4 bytes/float = **$4K$ bytes**.

**Total Bytes per data point** $= \mathbf{4(D + K)}$

---

## 3. The Arithmetic Intensity Formula
Combining the total work and the total traffic yields the Arithmetic Intensity per data point:

$$ \text{AI} = \frac{2 K D^2}{4(D + K)} = \mathbf{\frac{K D^2}{2(D + K)}} $$

---

## 4. Hardware Scaling Implications: The "K-Multiplier"

At first glance, one might assume that because Dimensionality ($D$) is squared, it is the primary driver of hardware scaling. However, the Arithmetic Intensity formula reveals that **Cluster Count ($K$) is the critical amortization factor**.

*   **Increasing $D$:** If we increase $D$, the numerator (work) grows as $D^2$, but the denominator (memory traffic) also grows linearly as $4D$. 
*   **Increasing $K$ (Data Reuse):** When we increase $K$, the numerator grows linearly, but the denominator $4(D+K)$ barely changes if $D$ is already large. 

**Conclusion:** 
$K$ represents **Temporal Locality** (Data Reuse). 
- At **low $K$**, we pay the high memory cost of fetching $x_n$ ($4D$ bytes) but perform very little math on it. The algorithm is strictly **Memory-Latency Bound** (favoring CPUs with large caches).
- At **high $K$**, we fetch $x_n$ once but reuse it $K$ times. We generate massive amounts of math ($2KD^2$) without requesting more data from RAM. The algorithm becomes **Compute-Throughput Bound**, which is precisely when the GPU's thousands of parallel cores overtake the CPU.
