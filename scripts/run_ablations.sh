#!/usr/bin/env bash
# Run all 5 ablation models sequentially in WSL.
# Each saves its checkpoint to checkpoints/ablation_<mode>/best_model.pth
#
# Usage (from WSL):
#   bash /mnt/e/deepfake-detection/scripts/run_ablations.sh
#
# Then compare results from logs/ablation_*/

set -e

source /home/noushad/miniconda3/etc/profile.d/conda.sh
conda activate oneclick
cd /home/noushad/deepfake-detection

DATA_DIR="/home/noushad/deepfake-detection/data"
CONFIG="configs/config.yaml"

MODES=(
    "spatial_only"
    "freq_only"
    "semantic_only"
    "spatial_freq"
    "spatial_semantic"
)

for MODE in "${MODES[@]}"; do
    echo "============================================"
    echo "ABLATION: $MODE"
    echo "============================================"
    python scripts/train.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --ablation-mode "$MODE" \
        2>&1 | tee "logs/train_ablation_${MODE}.log"
    echo "Done: $MODE"
done

echo ""
echo "All ablations complete. Checkpoints in checkpoints/ablation_*/"
echo "Run evaluate.py on each to compare AUC."
