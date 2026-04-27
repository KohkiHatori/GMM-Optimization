import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

def power_law(data, c, a, b):
    # data is (D, K)
    d, k = data
    return c * (d**a) * (k**b)

def analyze_scaling(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    d_data = df['D'].values
    k_data = df['K'].values
    y_data = df['WallTime'].values
    
    # Initial guess for [C, a, b]
    p0 = [0.001, 1.0, 1.0]
    
    try:
        popt, pcov = curve_fit(power_law, (d_data, k_data), y_data, p0=p0)
        c, a, b = popt
        
        print(f"\n--- Analysis for {label} ---")
        print(f"Mathematical Model: WallTime ≈ {c:.2e} * D^{a:.2f} * K^{b:.2f}")
        
        if a > b:
            print(f"RESULT: Dimensionality (D) has a higher structural impact (power = {a:.2f}).")
        else:
            print(f"RESULT: Cluster Count (K) has a higher structural impact (power = {b:.2f}).")
            
    except Exception as e:
        print(f"Error analyzing {label}: {e}")

# Run analysis
analyze_scaling("results/grid_benchmark/timing/serial.csv", "SERIAL")
analyze_scaling("results/grid_benchmark/timing/openmp_t32.csv", "OPENMP_T32")
analyze_scaling("results/grid_benchmark/timing/cuda.csv", "CUDA")
