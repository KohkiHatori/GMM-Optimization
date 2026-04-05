import argparse
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
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

def predict_clusters(X, weights, means, covariances):
    n, d = X.shape
    k = len(weights)
    
    log_resp = np.zeros((n, k), dtype=np.float32)
    
    for i in range(k):
        diff = X - means[i]
        reg_cov = covariances[i] + np.eye(d, dtype=np.float32) * 1e-4
        inv_cov = np.linalg.inv(reg_cov)
        sign, logdet = np.linalg.slogdet(reg_cov)
            
        mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        
        log_prob = -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)
        log_weights = np.log(np.clip(weights[i], 1e-15, 1.0))
        
        log_resp[:, i] = log_weights + log_prob
        
    return np.argmax(log_resp, axis=1)

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
    parser.add_argument("--true", type=str, required=True, help="Ground truth parameters .npz file")
    parser.add_argument("--init", type=str, required=True, help="Random initial means to sync validation states")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--title", type=str, default="GMM Clusters in 3D")
    parser.add_argument("--out_html", type=str, required=True, help="Output HTML filepath")
    args = parser.parse_args()
    
    if not os.path.exists(args.data) or not os.path.exists(args.our) or not os.path.exists(args.true):
        print("Error: Could not find input binaries for visualization.")
        exit(1)
        
    print(f"Loading data from {args.data}...")
    X = load_data(args.data, args.n, args.dim)
    
    # Load C parameters
    our_weights, our_means, our_covs = load_custom_model(args.our, args.k, args.dim)
    
    # Load Truth parameters
    true_data = np.load(args.true)
    true_means = true_data['means']
    true_labels = true_data['labels']
    
    print("Predicting Custom Cluster Allocations...")
    our_labels = predict_clusters(X, our_weights, our_means, our_covs)
    
    print("Fitting Scikit-Learn validation model (exact matched init)...")
    init_means = np.fromfile(args.init, dtype=np.float32).reshape(args.k, args.dim)
    init_weights = np.ones(args.k, dtype=np.float32) / args.k
    init_prec = np.array([np.eye(args.dim)] * args.k, dtype=np.float32)
    
    gmm = GaussianMixture(n_components=args.k, covariance_type='full', max_iter=200, tol=1e-4, random_state=42, 
                          means_init=init_means, weights_init=init_weights, precisions_init=init_prec, reg_covar=1e-4)
    sk_labels = gmm.fit_predict(X)
    sk_means = gmm.means_
    
    print("Permutation matching predicted categories to true categories...")
    # Remap our predictions to the matched true label ID
    matched_our = greedy_permutation_match(true_means, our_means)
    synced_our_preds = np.zeros_like(our_labels)
    for true_idx, our_idx in enumerate(matched_our):
        synced_our_preds[our_labels == our_idx] = true_idx
        
    # Remap SKLearn predictions to the matched true label ID
    matched_sk = greedy_permutation_match(true_means, sk_means)
    synced_sk_preds = np.zeros_like(sk_labels)
    for true_idx, sk_idx in enumerate(matched_sk):
        synced_sk_preds[sk_labels == sk_idx] = true_idx

    print("Squashing Dimensionality via PCA (D -> 3)...")
    if args.dim > 3:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
    elif args.dim == 3:
        X_3d = X
    else:
        # If D < 3, pad with zeros
        X_3d = np.zeros((args.n, 3))
        X_3d[:, :args.dim] = X

    # To optimize rendering for very large point clouds
    render_idx = np.arange(args.n)
    if args.n > 25000:
        print("N > 25000. Subsampling purely for browser rendering performance...")
        np.random.seed(42)
        render_idx = np.random.choice(args.n, 25000, replace=False)

    print("Rendering Plotly Side-by-Side 3D Interface...")
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Ground Truth", "Scikit-Learn (Random Init)", "Predicted (C EM)")
    )
    
    colors = px.colors.qualitative.Plotly
    symbols = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
    
    # 1. Truth
    for cluster_id in range(args.k):
        idx = (true_labels[render_idx] == cluster_id)
        if not np.any(idx): continue
        c, s = colors[cluster_id % len(colors)], symbols[cluster_id % len(symbols)]
        fig.add_trace(go.Scatter3d(
            x=X_3d[render_idx][idx, 0], y=X_3d[render_idx][idx, 1], z=X_3d[render_idx][idx, 2],
            mode='markers', marker=dict(size=3, color=c, symbol=s, opacity=0.7),
            name=f"True {cluster_id}", legendgroup=f"Group {cluster_id}"
        ), row=1, col=1)

    # 2. Scikit-Learn
    for cluster_id in range(args.k):
        idx = (synced_sk_preds[render_idx] == cluster_id)
        if not np.any(idx): continue
        c, s = colors[cluster_id % len(colors)], symbols[cluster_id % len(symbols)]
        fig.add_trace(go.Scatter3d(
            x=X_3d[render_idx][idx, 0], y=X_3d[render_idx][idx, 1], z=X_3d[render_idx][idx, 2],
            mode='markers', marker=dict(size=3, color=c, symbol=s, opacity=0.7),
            name=f"SK {cluster_id}", legendgroup=f"Group {cluster_id}", showlegend=False
        ), row=1, col=2)
        
    # 3. Predicted (C)
    for cluster_id in range(args.k):
        idx = (synced_our_preds[render_idx] == cluster_id)
        if not np.any(idx): continue
        c, s = colors[cluster_id % len(colors)], symbols[cluster_id % len(symbols)]
        fig.add_trace(go.Scatter3d(
            x=X_3d[render_idx][idx, 0], y=X_3d[render_idx][idx, 1], z=X_3d[render_idx][idx, 2],
            mode='markers', marker=dict(size=3, color=c, symbol=s, opacity=0.7),
            name=f"Pred {cluster_id}", legendgroup=f"Group {cluster_id}", showlegend=False
        ), row=1, col=3)
        
    fig.update_layout(
        title_text=args.title,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    fig.write_html(args.out_html)
    print(f"Visualization saved to {args.out_html}")
    
if __name__ == "__main__":
    main()
