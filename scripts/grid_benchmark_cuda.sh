#!/bin/bash

# Configuration
N=500000
D_ARRAY=(4 8 16 32)
K_ARRAY=(2 4 8 16 32 64 96)
ITERS=20
FIT_TOL=0.0001

# Directories
OUT_DIR="results/grid_benchmark/timing"
mkdir -p "$OUT_DIR"
CSV_OUT="$OUT_DIR/cuda.csv"

# Header
echo "N,D,K,WallTime" > "$CSV_OUT"

echo "Compiling..."
make cuda

for D in "${D_ARRAY[@]}"; do
    for K in "${K_ARRAY[@]}"; do
        echo "========================================"
        echo "GRID: N=$N, D=$D, K=$K (CUDA)"
        echo "========================================"

        DATA_DIR="data/grid_N${N}_D${D}_K${K}"
        DATA_BIN="${DATA_DIR}/data.bin"
        INIT_BIN="${DATA_DIR}/init_means.bin"

        # 1. Generate Data if missing
        if [ ! -f "$DATA_BIN" ]; then
            echo "-> Generating Data..."
            python3 data/generate_data.py --n $N --dim $D --components $K --out "$DATA_DIR"
        fi

        # 2. Run GMM
        echo "-> Running..."
        TEMP_LOG="/tmp/grid_cuda_$$.log"
        ./bin/gmm_cuda --data "$DATA_BIN" --init "$INIT_BIN" --n $N --dim $D --k $K --iters $ITERS --fit_tol $FIT_TOL > "$TEMP_LOG"

        # Extract Time
        WALL_TIME=$(awk -F': ' '/Total Wall Time/ {print $2}' "$TEMP_LOG" | awk '{print $1}')

        if [ ! -z "$WALL_TIME" ]; then
            echo "$N,$D,$K,$WALL_TIME" >> "$CSV_OUT"
            echo "Result: $WALL_TIME seconds"
        else
            echo "$N,$D,$K,ERROR" >> "$CSV_OUT"
        fi

        rm "$TEMP_LOG"
        echo ""
    done
done

echo "Grid benchmark complete. Results in $CSV_OUT"