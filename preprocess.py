import librosa
from librosa import feature
from librosa.core.convert import samples_like
import crepe
import numpy as np
from numpy import lib
import os

import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000

N_FFT = 1024
OVERLAP_RATE = 0.75
STEP_SIZE = int(N_FFT * (1 - OVERLAP_RATE))

N_MFCC = 40

CONF_THRESHOLD = 0.80
TARGET_LEN_FRAMES = 2500


class Preprocess():
    def __init__(self, audio_directory, output_directory, sample_per_output_file=20):
        self._sample_per_output_file = sample_per_output_file
        self._audio_features = {}

        self.process_directory(audio_directory, output_directory)
        

    def process_directory(self, audio_directory, output_directory):
        
        file_counter = 0
        output_counter = 1
        audio_features = {}
        signals = {}

        for file in os.listdir(audio_directory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav") or filename.endswith(".mp3"): 
                features, signal = self.process_audio_file(os.path.join(audio_directory, filename))

                audio_features[filename] = features
                signals[filename] = signal

                file_counter += 1
                if file_counter == self._sample_per_output_file:
                    self.save_to_npy(f"feature_{output_counter}", audio_features)
                    self.save_to_npy(f"signal_{output_counter}", signals)
                    output_counter += 1
                    file_counter = 0
                    signals.clear()
                    audio_features.clear()

    def process_audio_file(self, file_path):
        signal = self.read_audio(file_path, SAMPLE_RATE)
        features = self.extract_features(signal, SAMPLE_RATE)
        return features, signal



    def read_audio(self, file_path, targetSampleRate=SAMPLE_RATE):
        signal, _ = librosa.load(file_path, sr=targetSampleRate)
        return signal

    def save_to_npy(self, file_name, features):
        np.save(file_name, features, allow_pickle=True)
        

    # stacks the features in a 42 * # Frames matrix
    # (0) --- F0 ------------
    # (1) --- loudness ------
    # (2) --- First mfcc ----
    # (3) --- Second mfcc ---
    #  ......................
    # (42) --- 40th mfcc ----
    def extract_features(self, signal, sr):
        F0 = self.extract_F0(signal, sr)
        l = self.extract_loudness(signal, sr)
        mfcc = self.extract_MFCC(signal, sr)
        features = np.concatenate((np.array(F0).reshape(1, -1), l, mfcc), axis=0)
        return features


    def extract_F0(self, signal, sr):
        step_size_miliseconds = STEP_SIZE / SAMPLE_RATE * 1000
        time, frequency, confidence, activation = crepe.predict(signal, 
                                                                sr=sr, 
                                                                step_size=step_size_miliseconds, 
                                                                viterbi=True, 
                                                                verbose=0)
        
        indicies = np.where(confidence < CONF_THRESHOLD)[0]
        frequency[indicies] = 0.0

        return frequency


    def extract_loudness(self, signal, sr):
        s = librosa.stft(signal, n_fft=N_FFT, hop_length=STEP_SIZE)
        
        # Compute power and convert to dB scale.
        s = np.abs(s)
        s[s == 0] = 1e-20
        db = 20 * np.log10(s)

        # Weight
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        a_weighting = librosa.A_weighting(frequencies)
        a_weighting = a_weighting.reshape(-1, len(frequencies), 1)
        loudness = db + a_weighting


        # Set dynamic range.
        ref_db = 20.7
        range_db = 120.0

        loudness -= ref_db
        loudness = np.maximum(loudness, -range_db)

        loudness = np.mean(loudness, 1)
        return loudness


    def extract_MFCC(self, signal, sr):
        mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=N_MFCC, 
            n_fft=N_FFT, hop_length=STEP_SIZE, n_mels=128, fmin=20, fmax=8000)
        return mfcc


    def pad_or_trim_to_target(self):
        """Zero pad or trims the features to target number of frames. Pads and trims at the end of the array."""

        actual_num_frames = self._features.shape[1]
        if actual_num_frames > TARGET_LEN_FRAMES:
            self._features = self._features[: , : TARGET_LEN_FRAMES]
        elif actual_num_frames < TARGET_LEN_FRAMES:
            self._features = np.pad(self._features, pad_width=((0, 0), (0, TARGET_LEN_FRAMES - actual_num_frames)))


#prep = Preprocess("C:/Users/KXion/Desktop/flat", ".")

features = np.load("out.npy", allow_pickle=True)
print(features)

