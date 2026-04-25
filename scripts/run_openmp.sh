#!/bin/bash
# run_openmp.sh
# Builds the OpenMP GMM binary, sweeps thread counts x dataset sizes,
# validates against scikit-learn, generates cluster plots, and updates
# the benchmark scaling chart.

set -e

# Configuration Parameters
N_ARRAY=(10000 100000 500000 1000000 3000000)
THREAD_ARRAY=(1 2 4 8 16 24 32)
DIM=32
K=8
ITERS=50
FIT_TOL=0.0001
TOL=1.0

# Use .venv if present (Mac/local), otherwise fall back to system python3 (Linux/SCC)
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

echo "Compiling OpenMP binary..."
make clean
make openmp
echo ""

for T in "${THREAD_ARRAY[@]}"; do

    CSV_OUT="results/timing/openmp_t${T}.csv"
    echo "N,D,K,WallTime" > "$CSV_OUT"

    for N in "${N_ARRAY[@]}"; do
        echo "========================================"
        echo "N=$N | Threads=$T"
        echo "========================================"

        DATA_DIR="data/test_${N}"
        DATA_BIN="${DATA_DIR}/data.bin"
        INIT_BIN="${DATA_DIR}/init_means.bin"
        OUT_BIN="results/omp_out_${N}_t${T}.bin"
        TEMP_LOG="/tmp/gmm_omp_n${N}_t${T}_$$.log"

        # Generate data if missing
        if [ ! -f "$DATA_BIN" ]; then
            echo "-> Generating synthetic data for N=$N..."
            $PYTHON data/generate_data.py --n $N --dim $DIM --components $K --out "$DATA_DIR"
        fi

        # Run OpenMP GMM
        echo "-> Running OpenMP GMM (T=$T)..."
        export OMP_NUM_THREADS=$T
        ./bin/gmm_omp \
            --data "$DATA_BIN" \
            --init "$INIT_BIN" \
            --n $N --dim $DIM --k $K \
            --iters $ITERS --fit_tol $FIT_TOL \
            --out "$OUT_BIN" > "$TEMP_LOG"

        cat "$TEMP_LOG"

        # Extract wall time
        WALL_TIME=$(awk -F': ' '/Total Wall Time/ {print $2}' "$TEMP_LOG" | awk '{print $1}')
        if [ -n "$WALL_TIME" ]; then
            echo "$N,$DIM,$K,$WALL_TIME" >> "$CSV_OUT"
        else
            echo "$N,$DIM,$K,ERROR" >> "$CSV_OUT"
        fi

        # Validate against scikit-learn (only for T=1 to avoid redundant runs)
        if [ "$T" -eq 1 ]; then
            echo "-> Validating against scikit-learn..."
            $PYTHON validation/validate.py \
                --data "$DATA_BIN" \
                --our  "$OUT_BIN" \
                --init "$INIT_BIN" \
                --n $N --dim $DIM --k $K \
                --iters $ITERS --fit_tol $FIT_TOL \
                --tol $TOL

            echo "-> Generating 3D cluster plot..."
            $PYTHON visualization/plot_clusters.py \
                --data "$DATA_BIN" \
                --our  "$OUT_BIN" \
                --true "$DATA_DIR/true_params.npz" \
                --init "$INIT_BIN" \
                --n $N --dim $DIM --k $K \
                --title "OpenMP GMM Clusters (N=$N)" \
                --out_html "results/plots/omp_clusters_${N}.html"
        fi

        rm -f "$TEMP_LOG"
        echo ""
    done

    echo "Timing saved to $CSV_OUT"
    echo ""
done

# ── Regenerate the combined scaling chart (serial + all OMP thread counts) ─────
echo "========================================"
echo "Generating combined benchmark scaling chart"
echo "========================================"
$PYTHON visualization/plot_timings.py \
    --csv_dir results/timing/ \
    --out_html results/plots/execution_scaling.html

echo ""
echo "All done!"
echo "  Cluster plots : results/plots/omp_clusters_*.html"
echo "  Scaling chart : results/plots/execution_scaling.html"
echo "  Timing CSVs   : results/timing/openmp_t*.csv"
