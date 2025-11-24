#!/bin/bash

# HTCondor training script for GoNogo neurogym task
# Usage: ./train.sh MODEL_TYPE ETA LAMBDA HIDDEN_DIM NUM_EPISODES MAX_EPISODE_STEPS SUFFIX

set -e  # Exit on error

# Parse arguments
EXPERIMENT_NAME=$1
LATENT_SIZE=$2
BATCH_SIZE=$3
NUM_RESAMPLES=$4
EPOCHS=$5
LEARNING_RATE=$6
ALPHA=$7
NUM_TESTS=$8
    
# experiment_name, input_size, latent_size, batch_size, num_resamples, epochs, lr, alpha
 
# Generate experiment name: gonogo-model-verb-noun
EXP_NAME="VAE-IDS-${EXPERIMENT_NAME}"

# Print job information
echo "========================================"
echo "CSCI 597L VAE IDS Training Job"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Latent size: $LATENT_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Num resamples: $NUM_RESAMPLES"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Alpha: $ALPHA"
echo "Num tests: $NUM_TESTS"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "========================================"
echo ""

# Activate virtual environment
# conda activate cow

# Run training
echo "Starting training..."
../../anaconda3/bin/python ./main.py \
    --test-name ${EXP_NAME} \
    --latent-size ${LATENT_SIZE} \
    --num-resamples ${NUM_RESAMPLES} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --device cpu \
    --lr ${LEARNING_RATE} \
    --num-tests ${NUM_TESTS} \
    --alpha ${ALPHA} \

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"


exit $EXIT_CODE
