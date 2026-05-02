import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf

# --- Import your model classes ---
from src.models.conv import CONV
from src.models.conv_ssm import CONV_SSM_L
from src.models.gcn import GCN
from src.models.gcn_ssm import GCN_SSM


MODEL_REGISTRY = {
     "GCN": GCN, "GCN_SSM": GCN_SSM #......
}

class StreamingEvaluator:
    """
    Simulates real-time, block-based processing for a trained model.
    Maintains a state buffer to handle the model's receptive field.
    """
    def __init__(self, model, config):
        self.device = next(model.parameters()).device
        self.model = model
        self.config = config

        if not hasattr(model, 'calc_receptive_field'):
            raise NotImplementedError(
                f"Model {config['model']['name']} must have a 'calc_receptive_field()' method."
            )
        self.receptive_field = model.calc_receptive_field()
        print(f"Model receptive field: {self.receptive_field} samples")

        self.state_buffer_size = self.receptive_field - 1
        if self.state_buffer_size < 0: self.state_buffer_size = 0
            
        self.state_buffer = torch.zeros((self.state_buffer_size, 1), device=self.device)
        print(f"Initialized state buffer with size: {self.state_buffer_size} samples")

    def process_block(self, input_block):
        """
        Processes a single block of audio, managing the state buffer.
        
        Args:
            input_block (Tensor): Shape [Time, Channels=1]
        
        Returns:
            Tensor: Processed output block of the same shape.
        """
        if self.state_buffer_size == 0:
            model_input = input_block
        else:

            model_input = torch.cat([self.state_buffer, input_block], dim=0)


        model_input_batched = model_input.unsqueeze(0)
        
        # 2. Run inference on the full tensor
        with torch.no_grad():
            model_output_batched = self.model(model_input_batched)
        model_output = model_output_batched.squeeze(0)
        output_block = model_output[-input_block.shape[0]:, :]
        if self.state_buffer_size > 0:
            updated_buffer = torch.cat([self.state_buffer, input_block], dim=0)
 
            self.state_buffer = updated_buffer[-self.state_buffer_size:, :]
            
        return output_block

def run_streaming_test(checkpoint_path, input_path, output_path, block_size=256):
    # --- Load Model and Config ---
    ckpt_path = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config')
    model_name = config['model']['name']
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**config['model']['params']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    

    input_waveform, sr = torchaudio.load(input_path)
    target_sr = config['data']['sample_rate']
    if sr != target_sr:
        input_waveform = torchaudio.transforms.Resample(sr, target_sr)(input_waveform)
    if input_waveform.shape[0] > 1:
        input_waveform = torch.mean(input_waveform, dim=0, keepdim=True)
    
    # Reshape to [Time, Channels=1]
    input_waveform = input_waveform.permute(1, 0).to(device)

    # --- Initialize Streaming Evaluator ---
    evaluator = StreamingEvaluator(model, config)
    
    # --- Process Audio in Blocks ---
    num_blocks = int(np.ceil(input_waveform.shape[0] / block_size))
    output_chunks = []
    
    print(f"\nProcessing {input_waveform.shape[0]} samples in {num_blocks} blocks of size {block_size}...")
    for i in tqdm(range(num_blocks), desc="Streaming Inference"):
        start = i * block_size
        end = start + block_size
        input_chunk = input_waveform[start:end, :]
        
        # Pad the last block if necessary
        if input_chunk.shape[0] < block_size:
            pad_size = block_size - input_chunk.shape[0]
            input_chunk = torch.nn.functional.pad(input_chunk, (0, 0, 0, pad_size))
        
        output_chunk = evaluator.process_block(input_chunk)
        output_chunks.append(output_chunk)
        
    # --- Concatenate and Save Output ---
    full_output = torch.cat(output_chunks, dim=0)
    # Trim any padding we added to the last block
    full_output = full_output[:input_waveform.shape[0], :]
    
    output_waveform_final = full_output.permute(1, 0).cpu() # to [C, T]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), output_waveform_final.squeeze().numpy(), target_sr)
    print(f"\n Streaming inference complete. Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a model's real-time, block-based (streaming) capability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="Path to the .pth model checkpoint.")
    parser.add_argument("input", type=str, help="Path to the input audio file.")
    parser.add_argument("output", type=str, help="Path to save the processed output audio.")
    parser.add_argument("--block_size", type=int, default=256, help="The processing block size (buffer size).")
    args = parser.parse_args()

    run_streaming_test(args.checkpoint, args.input, args.output, args.block_size)