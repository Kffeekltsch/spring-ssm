import torch.optim as optim
import auraloss
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
#import torchaudio
from scipy.io.wavfile import write, read
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
#from SSM_velvet import SpringReverbNet_velvet, log_model_report

import librosa
import torchaudio
from torch.utils.data import Dataset

import torchaudio.functional as F_audio
from auraloss.utils import apply_reduction

class CompositeLoss(nn.Module):
    """
    Example composite loss: time-domain + STFT-based.
    """
    def __init__(self, device, sampling_rate, mel = True, alpha = 0.5, phase_weight = 0.0,w_log_mag = 1.0,output = "loss"): #alpha was 0.4
        super().__init__()
        self.time_loss = nn.L1Loss()
        self.freq_loss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes = [512, 1024, 2048, 4096], hop_sizes = [ 50, 100, 200, 400], win_lengths = [256, 512, 800, 1024],w_log_mag=w_log_mag,w_phs=phase_weight,output=output, perceptual_weighting = True,sample_rate=sampling_rate, device = device)
       # self.mel_loss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes = [128, 256, 512, 1024], hop_sizes = [25, 50, 100, 200], win_lengths = [128, 256, 512, 800], scale = 'mel', n_bins = 32, sample_rate=sampling_rate, device = device)
        self.mel_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],  # [ 128, 256, 512, 1024]
            hop_sizes=[50, 100, 200],        # [ 25, 50, 100, 200]
            win_lengths=[256, 512, 800],   # [128, 256, 512, 800]
            scale='mel',
            n_bins=32, #16 before
            perceptual_weighting = False, #new
            sample_rate=sampling_rate,
            device=device
        )
        #self.edc = EDCLoss()
        
        #self.mel_loss = auraloss.freq.MelSTFTLoss(sample_rate=44100,device="cuda")
        #self.wavelet = WaveletLoss(wave='db1', mode='zero', level=3)
        self.mae_loss = nn.MSELoss()
        self.alpha = alpha
        self.output = output
        self.skip_samples = 0 #500 before
        #self.freq_loss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 8192],hop_sizes=[256, 512, 2048],win_lengths=[1024, 2048, 8192],)
        #self.freq_loss = auraloss.freq.STFTLoss(fft_size = 2048, win_length = 2048) #change those for reverb tail eval?
        #self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512)
        #self.mel_loss = nn.L1Loss()
        #self.mel_loss_active = mel
    def forward(self, pred, target):
        # pred, target: shape (batch, 1, time)
        pred = pred[:,  self.skip_samples:, :]
        target = target[:, self.skip_samples:, :]
        loss_t = self.time_loss(pred, target)
        #loss_edc = self.edc(pred.permute(0,2,1), target.permute(0,2,1))
        if self.output == "full":
            loss_f,_,_,_, phs_loss = self.freq_loss(pred.permute(0,2,1), target.permute(0,2,1))
            #loss_phase = torch.mean(torch.stack(phs_loss))
        loss_f = self.freq_loss(pred.permute(0,2,1), target.permute(0,2,1))
        
        loss_mel = self.mel_loss(pred.permute(0,2,1), target.permute(0,2,1))
        #loss_wav = self.wavelet(pred.permute(0,2,1),target.permute(0,2,1))
        loss_mae = self.mae_loss(pred, target)
        loss_dc = dc_offset_loss(pred)
        """
        if self.mel_loss_active:
            pred_mel = self.mel_transform(pred.permute(0,2,1))
            target_mel = self.mel_transform(target.permute(0,2,1))
            loss_mel = self.mel_loss(pred_mel, target_mel)
        
            return loss_t + loss_f + 0.5 * loss_mel
        """
        
        return   loss_t + loss_f*(1-self.alpha) + loss_mel*self.alpha , loss_t, loss_f, loss_mel, loss_mae, loss_dc #0.2*loss_mae


class CompositetestLoss(nn.Module):
    """
    Example composite loss: time-domain + STFT-based.
    """
    def __init__(self,sampling_rate, device):
        super().__init__()
        self.time_loss = nn.L1Loss()
        self.freq_loss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes = [512, 1024, 2048, 4096], hop_sizes = [ 50, 100, 200, 400], win_lengths = [256, 512, 800, 1024], w_phs=0,output="loss", w_log_mag = 1.0, perceptual_weighting = True,sample_rate=sampling_rate, device = device)
        self.mae_loss = nn.MSELoss()
        self.esr = auraloss.time.ESRLoss()
        self.dc = auraloss.time.DCLoss()
        self.mel_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],  # [ 128, 256, 512, 1024]
            hop_sizes=[50, 100, 200],        # [ 25, 50, 100, 200]
            win_lengths=[256, 512, 800],   # [128, 256, 512, 800]
            scale='mel',
            n_bins=32, #16 before
            perceptual_weighting = False, #new
            sample_rate=sampling_rate,
            device=device
        )
        #self.freq_loss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 8192],hop_sizes=[256, 512, 2048],win_lengths=[1024, 2048, 8192],)
        #self.freq_loss = auraloss.freq.STFTLoss(fft_size = 2048, win_length = 2048) #change those for reverb tail eval?
        #self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512)
        #self.mel_loss = nn.L1Loss()
        #self.mel_loss_active = mel
    def forward(self, pred, target):
        # pred, target: shape (batch, 1, time)
        loss_t = self.time_loss(pred, target)
        loss_f = self.freq_loss(pred.permute(0,2,1), target.permute(0,2,1))
        loss_m = self.mae_loss(pred,target)
        loss_esr = self.esr(pred.permute(0,2,1), target.permute(0,2,1))
        loss_dc = self.dc(pred.permute(0,2,1), target.permute(0,2,1))
        loss_mel = self.mel_loss(pred.permute(0,2,1), target.permute(0,2,1))
        """
        if self.mel_loss_active:
            pred_mel = self.mel_transform(pred.permute(0,2,1))
            target_mel = self.mel_transform(target.permute(0,2,1))
            loss_mel = self.mel_loss(pred_mel, target_mel)
        
            return loss_t + loss_f + 0.5 * loss_mel
        """
        
        return loss_t, loss_f, loss_mel, loss_m, loss_esr, loss_dc