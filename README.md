# Hybrid Convolutional State-Space Models for Spring Reverb Emulation

This repository contains the official PyTorch implementation for the paper "Hybrid Convolutional State-Space Models for Spring Reverb Emulation".

We provide the code for our models (GCN_SSM, etc.), training and evaluation scripts, and a pretrained model checkpoint.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/spring-reverb-modeling.git
    cd spring-reverb-modeling
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Place your dry and wet audio files in the `data/EVT4500/` directory as follows:

- `data/EVT4500/dry/audio1.wav`
- `data/EVT4500/wet/audio1.wav`

The dataloader will automatically pair files with the same name.

## Training

To train a model, use the `train.py` script. You specify a model configuration file that inherits from the base configuration.

**Example: Train the `GCN_SSM` model**
```bash
python train.py --config configs/gcn_ssm.yaml --run_name "GCN_SSM_first_run" --notes "Training the full hybrid model."
