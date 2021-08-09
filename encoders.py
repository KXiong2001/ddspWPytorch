import crepe
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch.nn as nn
import torchaudio
import math  


F0_PREDICTION_CONFIDENCE_THRESHOLD = 0.8

class F0Encoder():
    def __init__(self):
        pass

    def __call__(self, signal, sr):
        time, frequency, confidence, activation = crepe.predict(signal, sr, viterbi=True)
        return frequency, confidence


class LEncoder():
    def __init__():
        pass

    def __call__(self, signal, sr):
        pass


MEL_SPEC_N_FFT = 1024
N_MFCC = 40

class ZEncoder(nn.Module):
    def __init__(self, signal_len, sr):
        super(ZEncoder, self).__init__()

        self._signal_len = signal_len
        self._sr = sr
        self._size_after_mfcc = math.ceil(self._signal_len / (MEL_SPEC_N_FFT // 4))

        melspec_args = {
            "n_fft": 1024,
            "hop_length": 1024 // 4,
            "f_min": 20,
            "f_max": 8000
        }

        self.mfcc = torchaudio.transforms.MFCC(sr, n_mfcc=N_MFCC, log_mels=True, melkwargs=melspec_args)
        
        # Normalization layer with learnable parameters
        # The input size ignores the batch size! [*, shape[0], shape[1] ... shape[-1]]
        layer_norm_input_shape = [N_MFCC, self._size_after_mfcc]
        self.layer_norm = nn.LayerNorm(layer_norm_input_shape)


        nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)
        """
        SIZE = 0
        hidden_size = 0

        pass"""

    def forward(self, signal):
        return self.layer_norm(self.mfcc(signal))


signal, sr = torchaudio.load("random_audio_for_testing3.wav", normalize=True)
print(signal.shape)
zEncoder = ZEncoder(signal.shape[1], sr)
print(zEncoder(signal).shape)


"""f0Encoder = F0Encoder()
f0, conf = f0Encoder(signal, sr)
print(len(signal), len(f0))"""