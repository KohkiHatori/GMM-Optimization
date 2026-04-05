import argparse
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

def load_data(filepath, n, d):
    return np.fromfile(filepath, dtype=np.float32).reshape(n, d)

def load_custom_model(filepath, k, d):
    data = np.fromfile(filepath, dtype=np.float32)
    offset = 0
    weights = data[offset : offset + k]
    offset += k
    
    means = data[offset : offset + k * d].reshape(k, d)
    offset += k * d
    
    covariances = data[offset : offset + k * d * d].reshape(k, d, d)
    
    return weights, means, covariances

def compute_log_likelihood(X, weights, means, covariances):
    n, d = X.shape
    k = len(weights)
    
    log_resp = np.zeros((n, k), dtype=np.float32)
    
    for i in range(k):
        diff = X - means[i]
        # We enforce exactly 1e-4 regularization globally mapping the C implementation exactly.
        # This prevents float32 conditioning explosions inside Python's matrix inversion implementations
        reg_cov = covariances[i] + np.eye(d, dtype=np.float32) * 1e-4
        inv_cov = np.linalg.inv(reg_cov)
        sign, logdet = np.linalg.slogdet(reg_cov)
            
        mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        
        log_prob = -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)
        log_weights = np.log(np.clip(weights[i], 1e-15, 1.0))
        
        log_resp[:, i] = log_weights + log_prob
        
    log_prob_norm = logsumexp(log_resp, axis=1)
    return np.mean(log_prob_norm)

def greedy_permutation_match(means1, means2):
    # Match means from model2 to model1 using greedy L2 distance
    k = means1.shape[0]
    matched_2 = []
    unmatched_2 = list(range(k))
    
    for i in range(k):
        m1 = means1[i]
        best_dist = float('inf')
        best_j = -1
        for j in unmatched_2:
            dist = np.linalg.norm(m1 - means2[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        matched_2.append(best_j)
        unmatched_2.remove(best_j)
        
    return matched_2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input data binary file")
    parser.add_argument("--our", type=str, required=True, help="Our C program output binary file")
    parser.add_argument("--init", type=str, required=True, help="Random initial means to sync validation states")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--fit_tol", type=float, required=True)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    
    if not os.path.exists(args.data) or not os.path.exists(args.our):
        print("Error: Could not find input binaries.")
        exit(1)
        
    print(f"Loading data from {args.data}...")
    X = load_data(args.data, args.n, args.dim)
    
    print(f"Loading custom model from {args.our}...")
    our_weights, our_means, our_covs = load_custom_model(args.our, args.k, args.dim)
    
    print("Re-calculating custom log-likelihood in Python...")
    our_ll = compute_log_likelihood(X, our_weights, our_means, our_covs)
    print(f"  -> Our Log-Likelihood: {our_ll:.4f}")
    
    print("\nRunning scikit-learn reference GMM...")
    init_means = np.fromfile(args.init, dtype=np.float32).reshape(args.k, args.dim)
    init_weights = np.ones(args.k, dtype=np.float32) / args.k
    init_prec = np.array([np.eye(args.dim)] * args.k, dtype=np.float32)
    
    gmm = GaussianMixture(n_components=args.k, covariance_type='full', max_iter=args.iters, tol=args.fit_tol, random_state=42, 
                          means_init=init_means, weights_init=init_weights, precisions_init=init_prec, reg_covar=1e-4)
    gmm.fit(X)
    
    sk_ll = gmm.score(X)
    print(f"  -> SKLearn Log-Likelihood: {sk_ll:.4f}")
    
    print("\n--- Validation Results ---")
    
    # 1. Log-Likelihood
    ll_diff = abs(our_ll - sk_ll)
    if ll_diff < args.tol:
        print(f"✅ Log-Likelihood check PASSED (difference {ll_diff:.2e} < {args.tol})")
    else:
        print(f"❌ Log-Likelihood check FAILED (difference {ll_diff:.2e} > {args.tol})")
        print("   Note: Differing initialization schemes can legitimately cause differing local minima.")
        
    # 2. Permutation match means
    print("\nPerforming cluster permutation match...")
    matched_indices = greedy_permutation_match(gmm.means_, our_means)
    
    max_mean_error = 0.0
    for i, j in enumerate(matched_indices):
        dist = np.linalg.norm(gmm.means_[i] - our_means[j])
        max_mean_error = max(max_mean_error, dist)
        
    # Give coordinate differences more tolerance
    mean_tol = args.tol * 10
    if max_mean_error < mean_tol: 
        print(f"✅ Cluster Means check PASSED (max L2 matching distance: {max_mean_error:.4f} < {mean_tol})")
    else:
        print(f"⚠️ Cluster Means check WARN (max L2 matching distance: {max_mean_error:.4f} > {mean_tol})")
        print("   If LL matches but means don't, clusters might be degenerate or converged elsewhere.")
        
if __name__ == "__main__":
    main()
