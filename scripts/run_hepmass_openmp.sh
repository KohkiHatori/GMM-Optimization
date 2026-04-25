#!/bin/bash

# Configuration Parameters
THREAD_ARRAY=(1 2 4 8 16 24)
DIM=28
K=8
ITERS=20
FIT_TOL=0.0001

# Directories
DATA_BIN="data/hepmass/data.bin"
INIT_BIN="data/hepmass/init_means.bin"

if [ ! -f "$DATA_BIN" ]; then
    echo "Error: HEPMASS data not found. Please run 'python3 data/prep_hepmass.py' first."
    exit 1
fi

mkdir -p results/timing results/plots

# Calculate N dynamically based on file size (D=28, float32=4 bytes)
N=$(python3 -c "import os; print(os.path.getsize('$DATA_BIN') // ($DIM * 4))")

echo "Compiling..."
make openmp

OUT_BIN="results/hepmass_omp_out.bin"

for T in "${THREAD_ARRAY[@]}"; do
    CSV_OUT="results/timing/hepmass_openmp_t${T}.csv"
    echo "N,D,K,WallTime" > "$CSV_OUT"
    
    echo "========================================"
    echo "Testing HEPMASS Data (N = $N, D = $DIM, K = $K) with Threads = $T"
    echo "========================================"

    export OMP_NUM_THREADS=$T
    
    # Run OpenMP GMM & Capture Output
    echo "-> Running OpenMP GMM..."
    TEMP_LOG="/tmp/temp_hepmass_omp_t${T}_$$.log"
    ./bin/gmm_omp --data "$DATA_BIN" --init "$INIT_BIN" --n $N --dim $DIM --k $K --iters $ITERS --fit_tol $FIT_TOL --out "$OUT_BIN" > "$TEMP_LOG"

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
done

echo "Done! Timing results saved to results/timing/"

echo "========================================"
echo "Generating Final Benchmark Scaling Graph"
echo "========================================"
python3 visualization/plot_timings.py --csv_dir results/timing/ --out_html results/plots/hepmass_timing_chart.html