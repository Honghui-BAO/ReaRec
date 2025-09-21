#!/bin/bash

# Example script to run ReaRec with LETTER data reader
# This script demonstrates how to use the migrated LETTER data reading functionality

echo "Running ReaRec with LETTER data reader..."

# Set parameters
MODEL_NAME="PRL"  # or "ERL"
DATASET="Beauty"  # or "Instruments", "Yelp"
GPU_ID="0"
USE_ITEM_FEATURES=1

# Run with LETTER reader
python src/main.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --use_letter_reader 1 \
    --use_item_features $USE_ITEM_FEATURES \
    --gpu $GPU_ID \
    --train 1 \
    --verbose 20

echo "Training completed!"
