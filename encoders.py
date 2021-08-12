import crepe
import torch.nn as nn
import torchaudio
import torch
import math
import librosa


N_FFT = 1024
OVERLAP_RATE = 0.75
N_MFCC = 40
hidden_size = 512 # 512-unit GRU
z_dimension = 16 # out feature size of the last dense layer

class F0Encoder():
    def __init__(self):
        pass

    def __call__(self, signal, sr):
        time, frequency, confidence, activation = crepe.predict(signal, sr, viterbi=True)
        return frequency, confidence


class LEncoder():
    def __init__(self):
        pass

    def __call__(self, signal, sr):
        hop_step = int(N_FFT * (1 - OVERLAP_RATE))

        s = torch.stft(signal, N_FFT, hop_length=hop_step, return_complex=True)
        
        # Compute power and convert to dB scale.
        s = torch.abs(s)
        s[s == 0] = 1e-20
        db = torch.mul(torch.log10(s), 20)

        # Weight
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        a_weighting = librosa.A_weighting(frequencies)
        a_weighting = a_weighting.reshape(-1, len(frequencies), 1)
        loudness = db + a_weighting


        # Set dynamic range.
        ref_db = 20.7
        range_db = 120.0

        loudness -= ref_db
        loudness = torch.maximum(loudness, torch.FloatTensor([-range_db]).expand_as(loudness))

        loudness = torch.mean(loudness, 1) # [batch, # frames]
                                        
        return loudness



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
        x = self.mfcc(signal)
        x = self.layer_norm(x) # [batch, # mfccs, # frames]
        
        x = x.transpose(1, 2) # [batch, # frames, # mfccs]
        x, h_n = self.gru(x) # [batch, # frams, # hidden_size(512)]
        x = self.dense(x)
        return x


class F0LZEncoder():
    def __init__():
        pass

    def __call__(self, signal, sr):
        pass


"""signal, sr = torchaudio.load("random_audio_for_testing3.wav", normalize=True)
l = LEncoder()
print(l(signal,sr))"""
"""print(signal.shape)
zEncoder = ZEncoder(signal.shape[1], sr)
print(zEncoder(signal).shape)"""
"""a"""

"""f0Encoder = F0Encoder()
f0, conf = f0Encoder(signal, sr)
print(len(signal), len(f0))"""