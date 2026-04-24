#!/bin/bash

# Configuration Parameters
N_ARRAY=(100000 500000 1000000)
DIM=32
K=8
ITERS=50
FIT_TOL=0.0001
TOL=1.0

# Directories
CSV_OUT="results/timing/serial.csv"
mkdir -p results/timing data scripts

# Setup CSV Header
echo "N,D,K,WallTime" > "$CSV_OUT"

echo "Compiling..."
make clean
make serial

for N in "${N_ARRAY[@]}"; do
    echo "========================================"
    echo "Testing N = $N"
    echo "========================================"

    DATA_DIR="data/test_${N}"
    DATA_BIN="${DATA_DIR}/data.bin"
    OUT_BIN="results/serial_out_${N}.bin"

    # 1. Generate Data
    if [ ! -f "$DATA_BIN" ]; then
        echo "-> Generating Synthetic Data..."
        python3 data/generate_data.py --n $N --dim $DIM --components $K --out "$DATA_DIR"
    fi

    # 2. Run GMM Serial & Capture Output
    echo "-> Running Serial GMM..."
    TEMP_LOG="/tmp/temp_serial_n${N}_$$.log"
    ./bin/gmm_serial --data "$DATA_BIN" --init "${DATA_DIR}/init_means.bin" --n $N --dim $DIM --k $K --iters $ITERS --fit_tol $FIT_TOL --out "$OUT_BIN" > "$TEMP_LOG"

    # Read the file so the user can see execution status live
    cat "$TEMP_LOG"

    # Extract Total Wall Time using awk
    WALL_TIME=$(awk -F': ' '/Total Wall Time/ {print $2}' "$TEMP_LOG" | awk '{print $1}')

    if [ ! -z "$WALL_TIME" ]; then
        echo "$N,$DIM,$K,$WALL_TIME" >> "$CSV_OUT"
    else
        echo "$N,$DIM,$K,ERROR" >> "$CSV_OUT"
    fi

    # 3. Run Validation
    echo "-> Validating against scikit-learn..."
    python3 validation/validate.py --data "$DATA_BIN" --our "$OUT_BIN" --init "${DATA_DIR}/init_means.bin" --n $N --dim $DIM --k $K --iters $ITERS --fit_tol $FIT_TOL --tol $TOL

    # 4. Generate 3D Interactive Plot
    echo "-> Generating 3D Plotly Visualization..."
    python3 visualization/plot_clusters.py --data "$DATA_BIN" --our "$OUT_BIN" --true "$DATA_DIR/true_params.npz" --init "${DATA_DIR}/init_means.bin" --n $N --dim $DIM --k $K --title "Serial GMM Clusters (N=$N)" --out_html "results/plots/serial_clusters_${N}.html"

    # Cleanup temp
    rm "$TEMP_LOG"
    echo ""
done

echo "Done! Timing results saved to $CSV_OUT"

echo "========================================"
echo "Generating Final Benchmark Scaling Graph"
echo "========================================"
python3 visualization/plot_timings.py --csv_dir results/timing/ --out_html results/plots/execution_scaling.html
