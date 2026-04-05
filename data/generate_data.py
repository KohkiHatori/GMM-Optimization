import os
import argparse
import numpy as np
from sklearn.datasets import make_blobs

def generate_gmm_data(n, dim, k, out_dir, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    print(f"Generating {n} points in {dim} dimensions with {k} clusters...")
    
    # 1. Generate isotropic Gaussian blobs
    X, y = make_blobs(n_samples=n, n_features=dim, centers=k, random_state=seed, center_box=(-20.0, 20.0))
    
    means = np.zeros((k, dim), dtype=np.float32)
    covariances = np.zeros((k, dim, dim), dtype=np.float32)
    weights = np.zeros(k, dtype=np.float32)
    
    # 2. Make them anisotropic (elliptical)
    # We apply a random affine transformation matrix to each cluster
    X_transformed = np.empty_like(X, dtype=np.float32)
    
    for i in range(k):
        # Indices of points belonging to cluster i
        idx = (y == i)
        cluster_points = X[idx]
        weights[i] = np.sum(idx) / n

        # Transformation matrix to stretch and rotate
        transformation = np.random.randn(dim, dim)
        
        # We shift points to the origin, transform, then shift back to a new spread out center
        center = np.mean(cluster_points, axis=0)
        shifted = cluster_points - center
        transformed = np.dot(shifted, transformation)
        
        # Nudge the new center to ensure decent spacing
        new_center = center + np.random.randn(dim) * 2.0
        X_transformed[idx] = transformed + new_center
        
        means[i] = new_center.astype(np.float32)
        
        # True covariance matrix = T^T * T
        cov = np.dot(transformation.T, transformation)
        covariances[i] = cov.astype(np.float32)
        
    # 3. Create outputs
    os.makedirs(out_dir, exist_ok=True)
    
    # 4. Save flat binary array formatting (float32)
    out_bin = os.path.join(out_dir, "data.bin")
    X_transformed.tofile(out_bin)
    print(f"[{os.path.getsize(out_bin) / 1024 / 1024:.3f} MB] Saved raw float32 data to: {out_bin}")
    
    # 5. Save the true parameters for future validation scripts
    out_npz = os.path.join(out_dir, "true_params.npz")
    np.savez(out_npz, means=means, covariances=covariances, weights=weights, labels=y)
    print(f"Saved true parameters (means, covariances, weights, labels) to: {out_npz}")
    
    # 6. Save random initial means to synchronize initialization across testing architectures
    init_idx = np.random.choice(n, k, replace=False)
    init_means = X_transformed[init_idx]
    out_init = os.path.join(out_dir, "init_means.bin")
    init_means.tofile(out_init)
    print(f"Saved random initial means to: {out_init}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic anisotropic data for GMM testing")
    parser.add_argument("--n", type=int, required=True, help="Number of samples")
    parser.add_argument("--dim", type=int, required=True, help="Number of dimensions")
    parser.add_argument("--components", type=int, required=True, help="Number of mixture components (K)")
    parser.add_argument("--out", type=str, required=True, help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generate_gmm_data(args.n, args.dim, args.components, args.out, seed=args.seed)
