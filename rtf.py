# measure_rtf.py
import argparse
from pathlib import Path
import yaml
import time
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import tqdm as tqdm_module


from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM

MODEL_REGISTRY = {
    "CONV": CONV, "CONV_SSM_L": CONV_SSM_L, "GCN": GCN, "GCN_SSM": GCN_SSM,
}

def measure_rtf(model, duration_s, sample_rate, num_runs=50, warmup_runs=10):
    """
    Measures the Real-Time Factor (RTF) of a model on a single CPU core.
    """
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    num_samples = int(duration_s * sample_rate)
    dummy_input = torch.randn(1, num_samples, 1, device=device)
    processing_times = []
    
    print(f"  - Measuring RTF over {num_runs} runs ({warmup_runs} warmup)...")
    with torch.no_grad():
        # Warmup runs
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            
        # Timed runs
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            processing_times.append(end_time - start_time)
            
    avg_time = np.mean(processing_times)
    rtf = avg_time / duration_s
    
    return rtf, avg_time

def analyze_config_rtf(config_path, output_dir):
    """
    Loads a model from a YAML config and analyzes its RTF.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(cfg_path, 'r') as f:
        model_config_update = yaml.safe_load(f)
    config.update(model_config_update)
    device = torch.device("cpu")
    model_name = config['model']['name']
    ModelClass = MODEL_REGISTRY[model_name]
    model_params = config['model']['params']
    
    model = ModelClass(**model_params).to(device)
    model.eval()
    print(f"\n--- Analyzing RTF for Model: {model_name} from {cfg_path.name} ---")
    
    data_cfg = config.get('data', {})
    sample_rate = data_cfg.get('sample_rate', 44100)
    duration = data_cfg.get('duration', 4.0)
    
    rtf, avg_time_s = measure_rtf(model, duration, sample_rate)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- RTF Summary ---")
    print(f"Model Name:         {model_name}")
    print(f"Trainable Params:   {trainable_params:,}")
    print(f"Avg Processing Time:{avg_time_s * 1000:.2f} ms (for a {duration}s chunk)")
    print(f"Real-Time Factor:   {rtf:.4f}")
    print("---------------------\n")
    
    return {
        'Model': model_name,
        'Params': trainable_params,
        'AvgTime_ms': avg_time_s * 1000,
        'RTF': rtf
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure the Real-Time Factor (RTF) for all model configs in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config_dir', 
        type=str, 
        default="configs",
        help="Directory containing the .yaml configuration files to analyze."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="final_paper_results",
        help="Directory to save the final RTF summary table."
    )
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    all_results = []
    config_files = [f for f in config_dir.glob("*.yaml") if f.name != "base.yaml"]
    
    for cfg_file in tqdm_module.tqdm(sorted(config_files), desc="Analyzing all configs"):
        result = analyze_config_rtf(cfg_file, args.output_dir)
        all_results.append(result)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.round({'AvgTime_ms': 2, 'RTF': 4})
        
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        save_path = output_path / "summary_rtf_and_params.csv"
        
        print("\n---Complexity and RTF Summary ---")
        print(summary_df.to_string(index=False))
        print("------------------------------------------")
        
        summary_df.to_csv(save_path, index=False)
        print(f"\n Summary table saved to: {save_path}")