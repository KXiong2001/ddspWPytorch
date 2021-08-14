import crepe
from scipy.io import wavfile

N_FFT = 1024
OVERLAP_RATE = 0.75

# ---- ZEncoder ---- #

N_MFCC = 40
hidden_size = 512 # 512-unit GRU
z_dimension = 16 # 
class F0Encoder():
    def __init__(self):
        pass

    def __call__(self, signal, sr):
        
        # signal: [# batch, # samples], torch Tensor

        step_size = int(N_FFT * (1 - OVERLAP_RATE)) / sr * 1000.0
        signal[signal == 0] = 1e-20
        time, frequency, confidence, activation = crepe.predict(signal, sr, viterbi=True)
        return frequency, confidence



sr, audio = wavfile.read('random_audio_for_testing3.wav',mmap=True)
f0Encoder = F0Encoder()
f0, conf = f0Encoder(audio, sr)