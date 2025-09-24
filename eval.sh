#!/bin/bash

# =============================================================================
# Evaluation Script for ICASSP Paper (Corrected Header)
# =============================================================================

# --- Configuration ---
CHECKPOINT_PARENT_DIR="final_checkpoints"
FINAL_RESULTS_DIR="final_icassp_eval"
DATA_DIR="data/final_dataset"
SUMMARY_CSV="${FINAL_RESULTS_DIR}/summary_all_models.csv"

set -e

if [ ! -d "$CHECKPOINT_PARENT_DIR" ]; then
    echo "❌ ERROR: Checkpoint parent directory not found at '${CHECKPOINT_PARENT_DIR}'"
    exit 1
fi

echo "================================================="
echo "🚀 Starting Final Evaluation Pipeline"
echo "Results will be saved to: ${FINAL_RESULTS_DIR}"
echo "================================================="

mkdir -p "$FINAL_RESULTS_DIR"

echo "Model,L1,MRSTFT,MelSpec_Loss,MSE,ESR_dB,Phase_Error" > "$SUMMARY_CSV"

for run_dir in "${CHECKPOINT_PARENT_DIR}"/*/; do
    
    checkpoint_path="${run_dir}checkpoint_best.pth"
    if [ ! -f "$checkpoint_path" ]; then
        echo "⚠️ WARNING: No 'checkpoint_best.pth' found in ${run_dir}. Skipping."
        continue
    fi
    
    model_name=$(basename "${run_dir}")
    
    echo ""
    echo "-------------------------------------------------"
    echo "Evaluating model: ${model_name}"
    echo "-------------------------------------------------"
    
    python evaluate.py \
        --checkpoint "${checkpoint_path}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${FINAL_RESULTS_DIR}" \
        --save_plots
        
    METRICS_FILE="${FINAL_RESULTS_DIR}/${model_name}/metrics_summary.csv"
    
    if [ -f "$METRICS_FILE" ]; then
        METRICS_LINE=$(tail -n 1 "$METRICS_FILE")
        echo "${model_name},${METRICS_LINE}" >> "$SUMMARY_CSV"
        echo "   -> Metrics collated successfully."
    else
        echo "⚠️ WARNING: Metrics summary file not found for ${model_name}. Skipping collation."
    fi
done

echo ""
echo "-------------------------------------------------"
echo "📊 Generating Final Comparison Plots..."
echo "-------------------------------------------------"

python plot_summary_results.py "${SUMMARY_CSV}" "${FINAL_RESULTS_DIR}"

echo ""
echo "================================================="
echo "🎉 All evaluations and plotting completed successfully!"
echo "Final results are in the '${FINAL_RESULTS_DIR}' directory."
echo "================================================="