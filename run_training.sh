#!/bin/bash

# =============================================================================
# Master Training Script for Spring Reverb Paper

# USAGE:chmod +x run_training.sh -> ./run_training.sh
# =============================================================================

# --- Configuration ---
# A unique name for this entire batch of experiments.
# All runs will be grouped under this name.
TIMESTAMP=$(date +"%Y%m%d_%H%M")
EXPERIMENT_GROUP="Final_Paper_Run_v01_${TIMESTAMP}"

# Array of the model config files to run.
MODELS_TO_TRAIN=(
    # --- CONV ---
    #"configs/conv_baseline.yaml"
    #"configs/conv_ssm_l_baseline.yaml" 
    # --- GCNs ---
    #"configs/gcn_baseline.yaml"
    #"configs/gcn_optimized.yaml"
    "configs/gcn_ssm_baseline.yaml"
    #"configs/gcn_ssm_optimized.yaml"
)


set -e 

echo "================================================="
echo "🚀 Starting Master Training Run"
echo "Experiment Group: ${EXPERIMENT_GROUP}"
echo "================================================="

# Loop through each model config and launch a job
for config_file in "${MODELS_TO_TRAIN[@]}"; do
    # Extract the model name from the filename (e.g., "gcn_ssm")
    model_name=$(basename "${config_file}" .yaml)
    
    # descriptive run name for wandb
    run_name="${model_name}_${TIMESTAMP}"
    
    echo ""
    echo "-------------------------------------------------"
    echo "Training model: ${model_name}"
    echo "Run Name: ${run_name}"
    echo "Config File: ${config_file}"
    echo "-------------------------------------------------"
    
    python train.py \
        --config "${config_file}" \
        --run_name "${run_name}" \

        
    echo "✅ Finished training for ${model_name}."
done

echo ""
echo "================================================="
echo "🎉 All training runs completed successfully!"
echo "Check your results under the group '${EXPERIMENT_GROUP}' in wandb."
echo "================================================="