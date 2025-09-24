# Hybrid Convolutional State-Space Models for Spring Reverb Emulation

This repository contains the official PyTorch implementation for the paper "Hybrid Convolutional State-Space Models for Spring Reverb Emulation".

We provide the code for our models (GCN_SSM, etc.), training and evaluation scripts, and a pretrained model checkpoint.

## Setup

1.  **Clone the repository:**


2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Place your dry and wet audio files in the `data' directory.
Update file names in eval.sh and train.sh

The dataloader will automatically pair files with the same name.

## Training

To train a model, use the `train.py` script. You specify a model configuration file that inherits from the base configuration.

**Example: Train the `GCN_SSM` model**
```bash
python train.py --config configs/gcn_ssm.yaml --run_name "GCN_SSM_first_run" --notes "Training the full hybrid model."
```

**Batch Training with Shell Script:**
For training multiple models sequentially, use the provided shell script:
```bash
chmod +x run_training.sh
./run_training.sh
```
Edit the `MODELS_TO_TRAIN` array in `run_training.sh` to select which models to train.

## Evaluation

### Model Evaluation
Evaluate a trained model on the test set:
```bash
python evaluate.py --checkpoint final_checkpoints/gcn-ssm-baseline/checkpoint_best.pth --data_dir data/EVT4500 --output_dir evaluation_results --save_plots
```

**Batch Evaluation with Shell Script:**
To evaluate all models in the `final_checkpoints/` directory:
```bash
chmod +x eval.sh
./eval.sh
```
This will generate a summary CSV file with metrics for all models.

## Inference

### Single Audio File Inference
Process a single audio file using a trained model:
```bash
python inference.py <checkpoint_path> <input_audio> <output_audio>
```

**Example:**
```bash
python inference.py final_checkpoints/gcn-ssm-baseline/checkpoint_best.pth audio_inference.wav gcn-ssm-stream.wav
```

**Optional parameters:**
- `--device`: Specify device (auto, cuda, cpu)
- `--no_norm`: Disable output normalization

### Streaming Inference
For real-time, block-based processing simulation:
```bash
python streaming_inference.py <checkpoint_path> <input_audio> <output_audio>
```

**Example:**
```bash
python streaming_inference.py final_checkpoints/gcn-ssm-baseline/checkpoint_best.pth audio_inference.wav gcn-ssm-stream-realtime.wav
```

This processes audio in chunks to simulate real-time processing with proper state management.

## Available Models

The repository includes several pre-trained models in `final_checkpoints/`:
- `conv-baseline/`: Convolutional baseline model
- `conv-ssm-baseline/`: Convolutional model with SSM layers
- `gcn-baseline/`: Graph Convolutional Network baseline
- `gcn-optimized/`: Optimized GCN model
- `gcn-ssm-baseline/`: Hybrid GCN-SSM baseline model
- `gcn-ssm-optimized/`: Optimized hybrid GCN-SSM model

## Additional Utilities

### Measuring Model Complexity
Calculate FLOPs and parameters for any model:
```bash
python measure_flops.py --config configs/gcn_ssm.yaml
```

### Real-Time Factor (RTF) Analysis
Measure processing speed relative to real-time:
```bash
python rtf.py final_checkpoints/gcn-ssm-baseline/checkpoint_best.pth
```

## File Structure

- `configs/`: Model configuration files
- `src/`: Source code for models, dataset, losses, and utilities
- `final_checkpoints/`: Pre-trained model checkpoints
- `audio_inference/`: Sample audio files for testing
- `*.sh`: Shell scripts for batch training and evaluation
- `inference.py`: Single file inference script
- `streaming_inference.py`: Real-time processing simulation
- `evaluate.py`: Model evaluation script
- `train.py`: Training script
