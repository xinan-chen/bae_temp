import re
from pathlib import Path
import random
import torch
from torch.utils import data
import numpy as np
from omegaconf import OmegaConf
# import simulate_utils as utils
import soundfile as sf
# from EQ_utils import LowPassFilter
import os
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

LABELPATH = '/data/ssd1/xinan.chen/VCTK/vctk-silence-labels/vctk-silences.0.92.txt'

class VctkDataset(data.Dataset):
    def __init__(
        self,
        dataset_dir,
        wav_len=4, # audio length in seconds
        sample_per_epoch=True, # whether to sample data per epoch
        num_per_epoch=10000,
        random_start=False, # whether to cut the audio from a random start point
        fs=16000,
        win_len=512,
        hop_len=256,
        fft_len=512,
        mode='train',
        low_fs=4000,
    ):
        super().__init__()
        assert mode in ['train', 'validation', 'test']

        self.dataset_dir = dataset_dir
        self.wav_len = wav_len * fs
        self.sample_per_epoch = sample_per_epoch
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        self.mode = mode

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len

        meta = os.listdir(os.path.join(dataset_dir, 'clean'))
        self.meta = list(filter(lambda x: x.endswith('wav'), meta))
        self.sample_data_per_epoch(mode) 

        self.timestamps = {}
        self.input_sample_rate = fs
        self.output_sample_rate = low_fs
        path_timestamps = LABELPATH
        with open(path_timestamps, 'r') as f:
            timestamps_list = f.readlines()
        for line in timestamps_list:
            timestamp_data = line.strip().split(' ')
            if len(timestamp_data) == 3:
                file_id, t_start, t_end = timestamp_data
                t_start = int(float(t_start) * self.input_sample_rate)
                t_end = int(float(t_end) * self.input_sample_rate)
                self.timestamps[file_id] = (t_start, t_end)

        # self.resample_forward = Resample(self.input_sample_rate, self.output_sample_rate)
        # self.resample_backward = Resample(self.output_sample_rate, self.input_sample_rate)

    def sample_data_per_epoch(self, mode='train'):
        if self.sample_per_epoch:
            if mode == 'train':
                self.meta_selected = random.sample(self.meta, self.num_per_epoch)
            else:
                self.meta_selected = self.meta[:self.num_per_epoch]
        else:
            self.meta_selected = self.meta

    def __getitem__(self, idx):
        # load files
        filename = self.meta_selected[idx]
        # noisy_path = os.path.join(self.dataset_dir, 'noisy', filename)
        clean_path = os.path.join(self.dataset_dir, 'clean', filename)
        # noisy, _ = sf.read(noisy_path, dtype='float32')
        try:
            clean, _ = sf.read(clean_path, dtype='float32')
        except Exception as e:
            print(f"Error reading file {clean_path}: {e}")
            clean = np.zeros(96000)

        # trim
        file_id = filename.split('_mic')[0]
        if file_id in self.timestamps:
            start, end = self.timestamps[file_id]
            start = start - min(start, int(0.1 * self.input_sample_rate))
            end = end + min(clean.shape[-1] - end, int(0.1 * self.input_sample_rate))
            clean = clean[start:end]
        clean = torch.tensor(clean, dtype=torch.float32)
        orig_len =clean.shape[-1]

        # select a clip with a fixed duration in seconds   
        if self.wav_len != 0:  # wav_len=0 means no cut or padding, use in valid/test
            if self.wav_len < orig_len:
                start_point = int(np.random.uniform(0, orig_len-self.wav_len)) if self.random_start else 0
                # noisy = noisy[start_point: start_point + self.wav_len]
                clean = clean[start_point: start_point + self.wav_len]
            elif self.wav_len > orig_len:
                pad_points = int(self.wav_len - orig_len)
                # noisy = F.pad(noisy, (0,pad_points), mode='constant', value=0)
                clean = F.pad(clean, (0,pad_points), mode='constant', value=0)

        # noisy = self.resample_backward(self.resample_forward(clean))
        # noisy = noisy[:self.wav_len]

        # normalization
        # std_ = torch.std(noisy, axis=-1, keepdims=True)
        # if std_ == 0:
        #     std_ = torch.tensor([1.0])
        #     print(f"std_ is zero in {filename}")
        # noisy = noisy / std_
        # clean = clean / std_

        # STFT
        # win = torch.hann_window(self.win_len)
        # clean_spec = torch.stft(clean, self.fft_len, self.hop_len, self.win_len, win, return_complex=True)  # (F,T)
        # low_freqbin = (self.win_len // 2) * self.output_sample_rate // self.input_sample_rate +1
        # noisy_spec = clean_spec.clone()
        # noisy_spec_mag = torch.abs(noisy_spec)
        # noisy_spec_mag[low_freqbin:,:] = 0
        # noisy = torch.istft(noisy_spec, self.fft_len, self.hop_len, self.win_len, win)
        # std_ = torch.std(noisy, axis=-1, keepdims=True)
        # if std_ == 0:
        #     std_ = torch.tensor([1.0])
        #     print(f"std_ is zero in {filename}")
        # clean_spec = clean_spec / std_ # (F,T)
        # noisy_spec = noisy_spec / std_
        # std_ = torch.tensor([1.0])
        # noisy_spec = torch.view_as_real(noisy_spec)  # (F,T,2)
        # clean_spec = torch.view_as_real(clean_spec)  # (F,T,2)

        return clean

    def __len__(self):
        return len(self.meta_selected)