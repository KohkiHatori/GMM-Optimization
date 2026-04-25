#!/bin/bash

# Configuration Parameters
DIM=28
K=8
ITERS=20
FIT_TOL=0.0001

# Directories
DATA_BIN="data/hepmass/data.bin"
INIT_BIN="data/hepmass/init_means.bin"
CSV_OUT="results/timing/hepmass_cuda.csv"

if [ ! -f "$DATA_BIN" ]; then
    echo "Error: HEPMASS data not found. Please run 'python3 data/prep_hepmass.py' first."
    exit 1
fi

mkdir -p results/timing results/plots

# Calculate N dynamically based on file size (D=28, float32=4 bytes)
N=$(python3 -c "import os; print(os.path.getsize('$DATA_BIN') // ($DIM * 4))")

# Setup CSV Header
echo "N,D,K,WallTime" > "$CSV_OUT"

echo "Compiling..."
make cuda

echo "========================================"
echo "Testing HEPMASS Data (N = $N, D = $DIM, K = $K) on CUDA"
echo "========================================"

OUT_BIN="results/hepmass_cuda_out.bin"

# Run CUDA GMM & Capture Output
echo "-> Running CUDA GMM..."
TEMP_LOG="/tmp/temp_hepmass_cuda_$$.log"
./bin/gmm_cuda --data "$DATA_BIN" --init "$INIT_BIN" --n $N --dim $DIM --k $K --iters $ITERS --fit_tol $FIT_TOL --out "$OUT_BIN" > "$TEMP_LOG"

# Read the file so the user can see execution status live
cat "$TEMP_LOG"

# Extract Total Wall Time using awk
WALL_TIME=$(awk -F': ' '/Total Wall Time/ {print $2}' "$TEMP_LOG" | awk '{print $1}')

if [ ! -z "$WALL_TIME" ]; then
    echo "$N,$DIM,$K,$WALL_TIME" >> "$CSV_OUT"
else
    echo "$N,$DIM,$K,ERROR" >> "$CSV_OUT"
fi

rm "$TEMP_LOG"
echo ""

echo "Done! Timing results saved to $CSV_OUT"

echo "========================================"
echo "Generating Final Benchmark Scaling Graph"
echo "========================================"
python3 visualization/plot_timings.py --csv_dir results/timing/ --out_html results/plots/hepmass_timing_chart.html