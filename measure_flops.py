# analyze_config.py
import argparse
from pathlib import Path
import yaml
import copy

import torch
import torch.nn as nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    print("WARNING: fvcore not found. FLOPs calculation will be skipped.")
    print("         Install with: pip install fvcore")
    FVCORE_AVAILABLE = False

# Import all models 
from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM

MODEL_REGISTRY = {
    "CONV": CONV, "CONV_SSM_L": CONV_SSM_L, "GCN": GCN, "GCN_SSM": GCN_SSM,
}

def get_cnn_flops(model, dummy_input):
    """
    Calculates FLOPs for only the non-SSM parts of a model by replacing
    SequenceLayer with nn.Identity. 
    """
    model_cnn_only = copy.deepcopy(model)
    
    # Recursively find and replace SequenceLayer modules
    def replace_ssm(module):
        for name, child in module.named_children():
            
            if "SequenceLayer" in child.__class__.__name__:
                setattr(module, name, nn.Identity())

            elif isinstance(child, (nn.ModuleList, nn.Sequential)):
                for i, sub_child in enumerate(child):
                    if "SequenceLayer" in sub_child.__class__.__name__:
                        child[i] = nn.Identity()
                    else:
                        replace_ssm(sub_child)
            else:
                replace_ssm(child)

    replace_ssm(model_cnn_only)

    try:
        flops_analyzer = FlopCountAnalysis(model_cnn_only, dummy_input)
        return flops_analyzer.total()
    except Exception as e:
        print(f"  - fvcore failed on CNN-only part: {e}")
        return 0

def analyze_config(config_path):
    """
    Loads a model from a YAML config and analyzes its complexity.
    """
    
    # 1. LOAD CONFIGURATION
   
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load base and model-specific configs and merge them
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(cfg_path, 'r') as f:
        model_config_update = yaml.safe_load(f)
    

    config.update(model_config_update)
    
    device = torch.device("cpu") # Analysis on CPU
    
    
    # 2. MODEL INSTANTIATION
    
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    
    ModelClass = MODEL_REGISTRY[model_name]
    model_params = config['model']['params']
    

    if 'sample_rate' in config['data']:
         if 'sample_rate' in ModelClass.__init__.__code__.co_varnames:
              model_params['sample_rate'] = config['data']['sample_rate']

    model = ModelClass(**model_params).to(device)
    model.eval()
    print(f"\n--- Analysis for Model: {model_name} from {cfg_path.name} ---")

    
    # 3. COMPLEXITY ANALYSIS
    
    data_cfg = config.get('data', {})
    sample_rate = data_cfg.get('sample_rate', 44100)
    duration = 1.0 #data_cfg.get('duration', 4.0)
    num_samples = int(sample_rate * duration)
    dummy_input = torch.randn(1, num_samples, model_params.get('in_channels', 1), device=device)
    
    print(f"Analyzing for a {duration}s input ({num_samples} samples) at {sample_rate} Hz...")

    # --- Calculate Parameters ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m_params = trainable_params / 1e6

    # --- Calculate FLOPs/MACs ---
    gflops = float('nan')
    analysis_method = "N/A"

    if FVCORE_AVAILABLE:
        try:
            flops_analyzer = FlopCountAnalysis(model, dummy_input)
            total_flops = flops_analyzer.total()
            analysis_method = "fvcore (automatic)"
        except Exception:
            print("Automatic analysis failed (expected for SSMs). Falling back to hybrid analysis.")
            cnn_flops = get_cnn_flops(model, dummy_input)
            
            ssm_macs = 0
            if hasattr(model, 'get_manual_macs'):
                ssm_macs = model.get_manual_macs(sequence_length=num_samples)
            
            total_flops = cnn_flops + (ssm_macs * 2)
            analysis_method = f"Hybrid (fvcore + manual)"
            print(f"  - CNN part (auto): {cnn_flops / 1e9:.3f} GFLOPs")
            print(f"  - SSM part (manual): {ssm_macs * 2 / 1e9:.3f} GFLOPs")
            
        gflops_per_sec = total_flops / duration / 1e9
    else:
        analysis_method = "fvcore not available"

    print("\n--- Model Complexity Summary ---")
    print(f"Trainable Params:   {m_params:.3f} M ({trainable_params:,} total)")
    print(f"GFLOPs/second:      {gflops_per_sec:.3f}")
    print(f"Analysis Method:    {analysis_method}")
    print("----------------------------------\n")

    return model_name, m_params, gflops_per_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model complexity from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'config_path', 
        type=str, 
        help="Path to the model's .yaml configuration file (e.g., configs/gcn_ssm.yaml)."
    )
    args = parser.parse_args()
    
    analyze_config(args.config_path)