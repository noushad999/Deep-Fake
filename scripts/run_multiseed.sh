#!/usr/bin/env bash
# Run training with seeds 123 and 777 for confidence intervals.
# Seed 42 already done: checkpoints/best_model.pth
# Results land in checkpoints/seed123/ and checkpoints/seed777/
set -e

source /home/noushad/miniconda3/etc/profile.d/conda.sh
conda activate oneclick
cd /home/noushad/deepfake-detection

DATA_DIR="/home/noushad/deepfake-detection/data"

for SEED in 123 777; do
    echo "============================================"
    echo "TRAINING SEED $SEED"
    echo "============================================"
    python scripts/train.py \
        --config "configs/config_seed${SEED}.yaml" \
        --data-dir "$DATA_DIR" \
        2>&1 | tee "logs/train_seed${SEED}.log"
    echo "Done: seed $SEED  →  checkpoints/seed${SEED}/best_model.pth"
done

echo ""
echo "Multi-seed training complete."
echo "Run eval on each checkpoint to get mean ± std."
