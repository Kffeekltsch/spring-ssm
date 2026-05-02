
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import soundfile as sf

from src.dataset import WavReverbDataset, split_dataset
from src.losses import CompositetestLoss
from src.utils.plotting import save_pcm_plot, save_melspectrogram_plot 
from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM



MODEL_REGISTRY = {
    "CONV": CONV,
    "CONV_SSM_L": CONV_SSM_L,
    "GCN": GCN,
    "GCN_SSM": GCN_SSM,
 
}

def calculate_mag_weighted_phase_error(target_stft, pred_stft, eps=1e-8):
    """Calculates magnitude-weighted phase error on torch tensors."""
    target_phase = torch.angle(target_stft)
    pred_phase = torch.angle(pred_stft)
    phase_diff = target_phase - pred_phase
    phase_diff_wrapped = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    abs_phase_error = torch.abs(phase_diff_wrapped)
    
    target_mag = torch.abs(target_stft)
    mag_weights = target_mag / (torch.sum(target_mag) + eps)
    
    return torch.sum(abs_phase_error * mag_weights).item()
    

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
    
    # 1. SETUP AND CHECKPOINT 
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

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
    

    config = checkpoint.get('config')
    if not config:
        print("Error: The checkpoint file does not contain a 'config' dictionary.")
        return
    print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'N/A')}. Evaluating on {device}.")
    model_name = config['model']['name']
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**config['model']['params']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model '{model_name}' loaded successfully.")

    data_cfg = config['data']
    duration = data_cfg.get('duration') or data_cfg.get('segment_duration')
    full_dataset = WavReverbDataset(
        dry_path=Path(data_dir) / 'dry_audio_final.wav',
        wet_path=Path(data_dir) / 'wet_audio_final.wav',
        sample_rate=config['data']['sample_rate'],
        duration=duration,
        load_in_memory=data_cfg.get('load_in_memory', False)
    )
    _, _, test_set = split_dataset(full_dataset)
    
    if num_samples:
        test_set = torch.utils.data.Subset(test_set, range(num_samples))
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(f"Evaluating on {len(test_set)} samples from the test set.")
    metrics_calculator = CompositetestLoss(sampling_rate=data_cfg['sample_rate'], device=device)
    results_list = []
    

    n_fft_phase = 1024
    hop_length_phase = 256
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluating {run_name}")
    with torch.no_grad():
        for i, (dry_wav, wet_wav) in pbar:
            original_index = test_set.indices[i] if isinstance(test_set, torch.utils.data.Subset) else i
            dry_wav, wet_wav = dry_wav.to(device), wet_wav.to(device)
            
            pred_wav = model(dry_wav)
            l1, mrstft, mel, mse, esr, _ = metrics_calculator(pred_wav, wet_wav)
            
            # Simple Phase Error
            wet_stft = torch.stft(wet_wav.squeeze(), n_fft=n_fft_phase, hop_length=hop_length_phase, return_complex=True)
            pred_stft = torch.stft(pred_wav.squeeze(), n_fft=n_fft_phase, hop_length=hop_length_phase, return_complex=True)
            phase_error = calculate_mag_weighted_phase_error(wet_stft, pred_stft)
            
            results_list.append({
                'sample_index': original_index, 'L1': l1.item(), 'MRSTFT': mrstft.item(),
                'MelSpec_Loss': mel.item(), 'MSE': mse.item(), 'ESR_dB': esr.item(), 
                'Phase_Error': phase_error
            })


            pred_np = pred_wav.squeeze().cpu().numpy()
            dry_np = dry_wav.squeeze().cpu().numpy()
            wet_np = wet_wav.squeeze().cpu().numpy()


            sf.write(audio_output_dir / f"sample_{original_index:03d}_pred.wav", pred_np, data_cfg['sample_rate'], subtype='PCM_24')
            sf.write(audio_output_dir / f"sample_{original_index:03d}_dry_ref.wav", dry_np, data_cfg['sample_rate'], subtype='PCM_24')
            sf.write(audio_output_dir / f"sample_{original_index:03d}_wet_ref.wav", wet_np, data_cfg['sample_rate'], subtype='PCM_24')

            if save_plots:
                tag = f"sample_{original_index:03d}"
                save_pcm_plot(dry_np, wet_np, pred_np, tag, output_dir=plots_output_dir)
                save_melspectrogram_plot(dry_np, wet_np, pred_np, data_cfg['sample_rate'], tag, output_dir=plots_output_dir)


    

    
    if not results_list:
        print("No samples were evaluated.")
        return

    #save per-sample results
    results_df = pd.DataFrame(results_list)
    detailed_csv_path = eval_output_dir / "metrics_per_sample.csv"
    results_df.to_csv(detailed_csv_path, index=False)
    print(f"\nSaved detailed metrics to {detailed_csv_path}")
    

    summary_metrics = results_df.mean().to_dict()
    column_order = [
        'L1',
        'MRSTFT',
        'MelSpec_Loss',
        'MSE',
        'ESR_dB',
        'Phase_Error'
    ]
    
    summary_df = pd.DataFrame([summary_metrics], columns=column_order)
    
    summary_csv_path = eval_output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n--- Evaluation Summary ---")
    print(summary_df.round(4).to_string(index=False))
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
        default="data", 
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