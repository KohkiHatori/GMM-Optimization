import pandas as pd
import numpy as np
import os

def prep_hepmass():
    out_dir = "data/hepmass"
    os.makedirs(out_dir, exist_ok=True)
    
    # Check for either the train or test file
    csv_path = os.path.join(out_dir, "1000_train.csv.gz")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(out_dir, "1000_test.csv.gz")
        
    bin_path = os.path.join(out_dir, "data.bin")
    init_path = os.path.join(out_dir, "init_means.bin")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find 1000_train.csv.gz or 1000_test.csv.gz in {out_dir}")
        print("Please download one manually using:")
        print(f"curl -L -o {out_dir}/1000_test.csv.gz https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_test.csv.gz")
        return

    print(f"Found dataset at {csv_path}")
    print(f"\nProcessing CSV into binary format (this might take a minute)...")
    
    # Read in chunks to save RAM
    chunksize = 1000000
    total_rows = 0
    dim = 0 
    
    with open(bin_path, 'wb') as f:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            # Column 0 is the label (Signal/Background). Columns 1-28 are features.
            features = chunk.iloc[:, 1:].values.astype(np.float32)
            dim = features.shape[1]
            features.tofile(f)
            total_rows += len(features)
            print(f"  -> Processed {total_rows:,} rows...")
            
    print(f"\n✅ Done! Saved {total_rows:,} points with {dim} dimensions to {bin_path}")
    
    # Generate random initial means for K=8 clusters
    print("\nGenerating random initial means for K=8...")
    df_sample = pd.read_csv(csv_path, nrows=5000)
    features_sample = df_sample.iloc[:, 1:].values.astype(np.float32)
    
    K = 8
    np.random.seed(42)
    init_idx = np.random.choice(len(features_sample), K, replace=False)
    init_means = features_sample[init_idx]
    init_means.tofile(init_path)
    print(f"✅ Saved initial means to {init_path}")
    
    print("\n" + "="*50)
    print("READY FOR DEMO!")
    print(f"Run the CUDA version on {total_rows:,} real physics events:")
    print(f"./bin/gmm_cuda --data {bin_path} --init {init_path} --n {total_rows} --dim {dim} --k {K} --iters 20")
    print("="*50)

if __name__ == "__main__":
    prep_hepmass()