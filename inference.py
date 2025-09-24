# inference.py
import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np


from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM



MODEL_REGISTRY = {
    "CONV": CONV, "CONV_SSM_L": CONV_SSM_L, "GCN": GCN, "GCN_SSM": GCN_SSM
}

def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    """Removes DC offset from a waveform tensor of shape [..., Time]."""
    return waveform - torch.mean(waveform, dim=-1, keepdim=True)

def run_inference(checkpoint_path, input_path, output_path, device_str='auto', no_norm=False):
    """
    Loads a trained model and processes a single audio file of any length.

    Args:
        checkpoint_path (str): Path to the .pth model checkpoint.
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the processed output audio file.
        device_str (str): Device to use ('cuda', 'cpu', or 'auto').
        no_norm (bool): If True, disables final peak normalization.
    """
 
    # 1. SETUP AND MODEL LOADING

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Loading checkpoint on {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config')
    if not config:
        raise ValueError("Checkpoint does not contain a 'config' dictionary.")

    model_name = config['model']['name']
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**config['model']['params']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model '{model_name}' loaded successfully.")


    # 2. AUDIO LOADING AND PREPROCESSING

    print(f"Loading audio from: {input_path}")
    input_waveform, sr = torchaudio.load(input_path)
    

    target_sr = config['data']['sample_rate']
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        input_waveform = resampler(input_waveform)

    # 2. Convert to mono
    if input_waveform.shape[0] > 1:
        print("Input is stereo, converting to mono...")
        input_waveform = torch.mean(input_waveform, dim=0, keepdim=True)
        
    # 3. Remove DC offset
    input_waveform = remove_dc_offset(input_waveform)

    
    model_input = input_waveform.permute(1, 0).unsqueeze(0).to(device)
    
    print(f"Prepared input tensor with shape: {model_input.shape}")

    
    # 3. INFERENCE
    
    print("Running inference...")
    with torch.no_grad():
        predicted_output = model(model_input)
    print("Inference complete.")
    
   
    # 4. POSTPROCESSING AND SAVING
   
    # Reshape back to audio format: [Batch, Time, Channels] -> [Channels, Time]
    output_waveform = predicted_output.squeeze(0).permute(1, 0).cpu()

    # Optional: Peak normalize the output to -0.1 dBFS
    if not no_norm:
        print("Peak normalizing output...")
        peak = torch.max(torch.abs(output_waveform))
        if peak > 1e-5: 
            output_waveform = output_waveform / peak * 0.99
    
   
    user_output_path = Path(output_path)
    
    
    directory = user_output_path.parent
    stem = user_output_path.stem     
    suffix = user_output_path.suffix   
    

    final_filename = f"{stem}_{model_name}{suffix}"
    final_output_path = directory / final_filename
    

    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save audio file
    print(f"Saving processed audio to: {final_output_path}")
    torchaudio.save(
        str(final_output_path),
        output_waveform,
        sample_rate=target_sr,
        encoding="PCM_S",   
        bits_per_sample=24 
    )
    print(" Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a single audio file with a trained spring reverb model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint", 
        type=str, 
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to the input audio file (.wav, .flac, etc.)."
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Path to save the processed output audio file."
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Device to run inference on."
    )
    parser.add_argument(
        '--no_norm', 
        action='store_true', 
        help="Disable final peak normalization of the output audio."
    )
    args = parser.parse_args()

    run_inference(args.checkpoint, args.input, args.output, args.device, args.no_norm)