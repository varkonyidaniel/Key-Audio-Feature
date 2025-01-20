import numpy as np
import librosa
from Enum import enum_types as et


def Fit(data: np.ndarray, type: str, hop_length: int, n_features: int, sampling_rate: int, n_fft: int=10):

        # determine return matrix dimensions:
        # dim[0]: n_features (CHROMA:12, SC:1, ZCR:1)
        # dim[1]: math.floor([total_length / hop_length ]+ 1)

        if type == et.Feature_type.STFT:
            return librosa.stft(y=data, n_fft=(n_features-1)*2, hop_length=hop_length, window='hann')
        elif type == et.Feature_type.MFCC:
            return librosa.feature.mfcc(y=data, sr=sampling_rate,n_mfcc=n_features,hop_length=hop_length,n_fft=512)
        elif type == et.Feature_type.CHROMA:
            return librosa.feature.chroma_stft(y=data, sr=sampling_rate, n_fft=n_features, hop_length=hop_length)
        elif type == et.Feature_type.ZCR:
            return librosa.feature.zero_crossing_rate(y=data, hop_length = hop_length, center=True, pad=False)
        elif type == et.Feature_type.SC:
            return librosa.feature.spectral_centroid(y=data, sr=sampling_rate,hop_length=hop_length, n_fft = n_fft)
        elif type == et.Feature_type.MFCC_DELTA:
            mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_features, hop_length=hop_length, n_fft = 512)
            return librosa.feature.delta(data = mfcc, width=3, order=1)
        else:
            return data

# sinus transformation, cosinus transformation, walsh- Hadamard Transform
# kevert transzformációk (sin bázisból 1 ki és helyette 1 cosinusz elem....)
