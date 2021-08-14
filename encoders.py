import torch.nn as nn
import torchaudio
import torch
import math
import librosa

from test import F0Encoder


N_FFT = 1024
OVERLAP_RATE = 0.75

# ---- ZEncoder ---- #

N_MFCC = 40
hidden_size = 512 # 512-unit GRU
z_dimension = 16 # out feature size of the last dense layer


class ZEncoder(nn.Module):
    def __init__(self, signal_len, sr):
        super(ZEncoder, self).__init__()

        hop_step = int(N_FFT * (1 - OVERLAP_RATE))

        self._signal_len = signal_len
        self._sr = sr
        self._size_after_mfcc = math.ceil(self._signal_len / hop_step)

        melspec_args = {
            "n_fft": 1024,
            "hop_length": hop_step,
            "f_min": 20,
            "f_max": 8000
        }

        self.mfcc = torchaudio.transforms.MFCC(sr, n_mfcc=N_MFCC, log_mels=True, melkwargs=melspec_args)
        
        # Normalization layer with learnable parameters
        # The input size ignores the batch size! [*, shape[0], shape[1] ... shape[-1]]
        layer_norm_input_shape = [N_MFCC, self._size_after_mfcc]
        self.layer_norm = nn.LayerNorm(layer_norm_input_shape)


        self.gru = nn.GRU(N_MFCC, hidden_size, batch_first=True)
        self.dense = nn.Linear(in_features=hidden_size, out_features=z_dimension)

    def forward(self, signal):

        # Signal: [#batch, #samples]

        x = self.mfcc(signal)
        x = self.layer_norm(x) # [batch, # mfccs, # frames]
        
        x = x.transpose(1, 2) # [batch, # frames, # mfccs]
        x, h_n = self.gru(x) # [batch, # frams, # hidden_size(512)]
        x = self.dense(x) # [batch, # frames, # Z dimension (16)]
        return x


signal, sr = torchaudio.load("random_audio_for_testing3.wav", normalize=True) # [channel, time]

# vvvv To mimic the data comes in batches! vvvv #

print(signal.shape)
signal = signal.unsqueeze(0)
signal = torch.cat((signal, signal), 0)
signal = signal.squeeze(1) # squeeze out the channel, assuming the signal is monochannel

# ^^^^ ----------------------------------- ^^^^ #

"""print(signal.shape)
l = LEncoder()
print(l(signal,sr))"""


"""print(signal.shape)
zEncoder = ZEncoder(signal.shape[1], sr)
print(zEncoder(signal).shape)"""

f0Encoder = F0Encoder()
f0 = f0Encoder(signal, sr)
print(len(signal), len(f0))