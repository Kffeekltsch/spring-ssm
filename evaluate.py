# evaluate.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import soundfile as sf

# Local module imports from the 'src' directory
from src.dataset import WavReverbDataset, split_dataset
from src.losses import TestMetrics
from src.utils.plotting import save_pcm_plot, save_spectrogram_plot # Assumes this file exists

# Import all models to create a registry (same as in train.py)
from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM

# Model registry to map config names to model classes
MODEL_REGISTRY = {
    "CONV": CONV,
    "CONV_SSM_L": CONV_SSM_L,
    "GCN": GCN,
    "GCN_SSM": GCN_SSM,
}


def evaluate(checkpoint_path, data_dir, output_dir, num_samples=None, save_plots=False):
    """
    Evaluates a trained model from a local checkpoint file.

    
    1. Load a model and its configuration from a .pth checkpoint file.
    2. Load the test portion of a specified dataset.
    3. Run inference for each sample in the test set.
    4. Calculate a suite of objective metrics.
    5. Save the predicted audio, reference audio, plots, and a final metrics report.

    Args:
        checkpoint_path (str): Path to the model checkpoint file (.pth).
        data_dir (str): Path to the root of the dataset directory (e.g., 'data/EVT4500').
        output_dir (str): Path to save the evaluation results.
        num_samples (int, optional): Limit evaluation to the first N samples for a quick test.
        save_plots (bool, optional): If True, generate and save plots for each sample.
    """
    
    # 1. SETUP AND CHECKPOINT LOADING
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    # Use the checkpoint's parent directory name to name the output folder
    run_name = ckpt_path.parent.name
    eval_output_dir = Path(output_dir) / run_name
    audio_output_dir = eval_output_dir / "audio"
    plots_output_dir = eval_output_dir / "plots"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plots_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load config from the checkpoint itself
    config = checkpoint.get('config')
    if not config:
        print("Error: The checkpoint file does not contain a 'config' dictionary.")
        print("Please retrain the model with the updated train.py script to embed the config.")
        return
    print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'N/A')}. Evaluating on {device}.")

    
    # 2. MODEL AND DATA LOADING
    
    # Instantiate model from the config saved in the artifact
    model_name = config['model']['name']
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**config['model']['params']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model '{model_name}' loaded successfully.")

    # Load test data
    full_dataset = WavReverbDataset(
        dry_dir=Path(data_dir) / 'dry',
        wet_dir=Path(data_dir) / 'wet',
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['segment_duration']
    )
    _, _, test_set = split_dataset(full_dataset)
    
    if num_samples:
        test_set = torch.utils.data.Subset(test_set, range(num_samples))
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(f"Evaluating on {len(test_set)} samples from the test set.")

    
    # 3. EVALUATION LOOP
    
    metrics_calculator = TestMetrics(sample_rate=config['data']['sample_rate'], device=device)
    results_list = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluating {run_name}")
    with torch.no_grad():
        for i, (dry_wav, wet_wav) in pbar:
            original_index = test_loader.dataset.indices[i] if isinstance(test_loader.dataset, torch.utils.data.Subset) else i

            dry_wav, wet_wav = dry_wav.to(device), wet_wav.to(device)
            
            # Run inference
            pred_wav = model(dry_wav)
            
            # Calculate metrics
            l1, mrstft, mel, mse, esr, dc = metrics_calculator(pred_wav, wet_wav)
            
            results_list.append({
                'original_sample_index': original_index,
                'L1': l1.item(),
                'MRSTFT': mrstft.item(),
                'MelSpec_Loss': mel.item(),
                'MSE': mse.item(),
                'ESR_dB': esr.item(),
                'DC_Offset': dc.item(),
            })

            # --- Save audio files ---
            pred_np = pred_wav.squeeze().cpu().numpy()
            dry_np = dry_wav.squeeze().cpu().numpy()
            wet_np = wet_wav.squeeze().cpu().numpy()

            sf.write(audio_output_dir / f"sample_{original_index:03d}_pred.wav", pred_np, config['data']['sample_rate'])
            sf.write(audio_output_dir / f"sample_{original_index:03d}_dry_ref.wav", dry_np, config['data']['sample_rate'])
            sf.write(audio_output_dir / f"sample_{original_index:03d}_wet_ref.wav", wet_np, config['data']['sample_rate'])

            # --- Optionally save plots ---
            if save_plots:
                tag = f"sample_{original_index:03d}"
                # You'd need to implement these plotting functions in src/utils/plotting.py
                # save_pcm_plot(dry_np, wet_np, pred_np, tag, output_dir=plots_output_dir)
                # save_spectrogram_plot(dry_np, wet_np, pred_np, config['data']['sample_rate'], tag, output_dir=plots_output_dir)

    
    # 4. AGGREGATE AND SAVE RESULTS
    
    if not results_list:
        print("No samples were evaluated.")
        return

    # Create and save detailed per-sample results
    results_df = pd.DataFrame(results_list)
    detailed_csv_path = eval_output_dir / "metrics_per_sample.csv"
    results_df.to_csv(detailed_csv_path, index=False)
    print(f"\nSaved detailed metrics for {len(results_df)} samples to {detailed_csv_path}")
    
    # Calculate and save summary statistics
    summary_df = results_df.mean().to_frame().T.drop(columns=['original_sample_index'])
    summary_csv_path = eval_output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n--- Evaluation Summary ---")
    print(summary_df.round(4))
    print("--------------------------")
    print(f"Evaluation finished. Results saved in: {eval_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained spring reverb model from a local checkpoint.")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True, 
        help="Path to the model checkpoint file (.pth) saved by the training script."
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default="data/EVT4500", 
        help="Path to the root of the dataset directory containing 'dry' and 'wet' subfolders."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="evaluation_results", 
        help="Directory to save all evaluation outputs."
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        help="Optional: limit evaluation to the first N test samples for a quick check."
    )
    parser.add_argument(
        '--save_plots', 
        action='store_true', 
        help="Generate and save diagnostic plots for each evaluated sample."
    )
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.output_dir, args.num_samples, args.save_plots)