import librosa
from librosa.core.convert import samples_like
import crepe
import numpy as np
from numpy import lib

SAMPLE_RATE = 16000

N_FFT = 1024
OVERLAP_RATE = 0.75
STEP_SIZE = int(N_FFT * (1 - OVERLAP_RATE))

N_MFCC = 40

CONF_THRESHOLD = 0.80

class Preprocess():
    def __init__(self, filePath):
        self._signal = self.read_audio(filePath, SAMPLE_RATE)
        self.extract_features()
        self.save_features()
        

    def read_audio(self, filePath, targetSampleRate=16000):
        signal, _ = librosa.load(filePath, sr=targetSampleRate)
        return signal

    def save_features(self):
        savefile = {
            'aaa': self._features
        }
        np.savez("out.npz", savefile)
        
    def extract_features(self):
        self._F0 = self.extractF0()
        self._L = self.extractLoudness()
        self._mfcc = self.extractMFCC()
        self._features = np.concatenate((np.array(self._F0).reshape(1, -1), self._L, self._mfcc), axis=0)

    def extractF0(self):
        step_size_miliseconds = STEP_SIZE / SAMPLE_RATE * 1000
        time, frequency, confidence, activation = crepe.predict(self._signal, 
                                                                sr=SAMPLE_RATE, 
                                                                step_size=step_size_miliseconds, 
                                                                viterbi=True, 
                                                                verbose=0)
        
        indicies = np.where(confidence < CONF_THRESHOLD)[0]
        frequency[indicies] = 0.0
        print("frequency ", frequency.shape)

        return frequency

    def extractLoudness(self):
        s = librosa.stft(self._signal, n_fft=N_FFT, hop_length=STEP_SIZE)
        
        # Compute power and convert to dB scale.
        s = np.abs(s)
        s[s == 0] = 1e-20
        db = 20 * np.log10(s)

        # Weight
        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        a_weighting = librosa.A_weighting(frequencies)
        a_weighting = a_weighting.reshape(-1, len(frequencies), 1)
        loudness = db + a_weighting


        # Set dynamic range.
        ref_db = 20.7
        range_db = 120.0

        loudness -= ref_db
        loudness = np.maximum(loudness, -range_db)

        loudness = np.mean(loudness, 1)
        print("loudness ", loudness.shape)
        return loudness


    def extractMFCC(self):
        mfcc = librosa.feature.mfcc(self._signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
            n_fft=N_FFT, hop_length=STEP_SIZE, n_mels=128, fmin=20, fmax=8000)
        return mfcc

prep = Preprocess("random_audio_for_testing3.wav")
