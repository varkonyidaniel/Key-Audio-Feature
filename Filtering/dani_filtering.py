import os, sys
from scipy.io import wavfile
from librosa import stft,istft
from librosa.util import fix_length
from scipy.stats import zscore
from scipy.fft import fft, ifft
#from Enum import Enum_types as et
from Enum import enum_types as et
import numpy as np
from scipy import signal
import Outputs.Draw as d
import librosa



class dani_filtering:

    def __init__(self,separation_logic:int=et.separation_type.BY_TIME_SLICES.value):
        self.separation_logic = et.separation_type(separation_logic)
        self.axis = separation_logic
        if separation_logic == 2:
            self.axis = None


    def norming(self,data:np.ndarray, norming_type:et.norming_type.Z_NORMING):

        tmp = zscore(data, axis=self.axis)
        N = np.size(data,self.axis)
        if norming_type == et.norming_type.SCALING:
            tmp*=np.sqrt((N-1)/N)
        return tmp


    def tresholding(self,data:np.ndarray, filtering_type:et.dani_filtering_type=et.dani_filtering_type.NEGATIVE):
        if filtering_type == et.dani_filtering_type.NEGATIVE:
            return np.where(data < 0, 0, data)
        else: # MEAN by separation
            avg = data.mean(axis=self.axis)

            if self.axis == 1:
                return np.where(data.T<avg.T,0,data.T).T
            else:
                return np.where(data < avg, 0, data)


    def limited_fft(self, data:np.ndarray, num_of_fft_components:int=50):
        resh = False

        if self.separation_logic == et.separation_type.WHOLE_ONE_UNIT:
            resh = True
            shape = data.shape

        tmp = fft(x=data.ravel() if resh else data, axis=0 if self.axis == None else self.axis,
                  norm = None, overwrite_x=False)

        if self.separation_logic == et.separation_type.WHOLE_ONE_UNIT:
            tmp[num_of_fft_components:]=0
        elif self.separation_logic == et.separation_type.BY_TIME_SLICES:
            tmp[num_of_fft_components:] = 0
        else:
            tmp[:,num_of_fft_components:] = 0

        tmp = ifft(tmp, axis=0 if self.axis == None else self.axis, norm=None, overwrite_x=False)

        if self.separation_logic == et.separation_type.WHOLE_ONE_UNIT:
            tmp = np.reshape(tmp,shape)
        return tmp


    def visualize(self,data,idx,xlabel, ylabel,title,title_add, draw_ts,draw_hm, xvalues, yvalues,sav_fig, dpi ):
        if draw_ts:
            if self.separation_logic == et.separation_type.BY_TIME_SLICES:
                d.drawTimeSeries(data=data[:,idx], xlabel=xlabel, ylabel=ylabel,
                                 text=str(title)+title_add,save_fig=sav_fig,xvalues=yvalues,dpi=dpi)
            else:
                d.drawTimeSeries(data=data[idx], xlabel=xlabel, ylabel=ylabel,
                                 text=title+title_add,save_fig=sav_fig, xvalues=xvalues,dpi=dpi)
        if draw_hm:
            d.drawHeatMap(data=data, xvalues=xvalues,yvalues=yvalues,xlabel='Time (sec)', ylabel='Frequvency (Hz)',
                          title=title,save_fig=sav_fig,dpi=dpi)


    def fit_demo(self, wav_data:np.ndarray, sr:int = 22050, freq_id:int = 199, draw_ts:bool=False,
                 draw_hm:bool=False, num_of_fft_components:float=0.5,save_fig:bool=False, dpi:int=300):
        return self.fit(data=wav_data, sr=sr, freq_id=freq_id, dr_ts=draw_ts, dr_hm=draw_hm,
                        num_of_fft_components=num_of_fft_components, save_fig=save_fig,dpi=dpi)


    def fit(self,data:np.ndarray, sr:float = 8000, freq_id:int = 130,dr_ts:bool=False,
            dr_hm:bool=False, num_of_fft_components:float=0.5,save_fig:bool=False,dpi:int=300):

        len_dim_1 = len(data)       # frequencies - 10
        len_dim_2 = len(data[0])    # times - 657

        if len_dim_1 == 1:      # SC, ZCR case: only 1 frequency
            frequencies = 1
        else:
            frequencies = np.arange(0, len_dim_1) * sr / ((len_dim_1 - 1) * 2)


        times = np.linspace(0,len_dim_2/sr, len_dim_2+1)

        if self.separation_logic == et.separation_type.BY_TIME_SLICES:
            lab_x = "Frequency (Hz)"
            lab_y = "Magnitude"
            ts_id = min(freq_id, len_dim_2 - 1)
            tx2 = f" at {round(times[ts_id],2)} min"
            n_of_fft = int(num_of_fft_components * len_dim_1)
        else:
            lab_x = "Time"
            lab_y = "Amplitude"
            ts_id = min(freq_id, len_dim_1 - 1)
            if len_dim_1 == 1:          # sc, zcr
                tx2 = frequencies
            else:
                tx2 = f" at {int(frequencies[ts_id])} Hz"
            n_of_fft = int(num_of_fft_components * len_dim_2)

        tx = 'Original data'
        self.visualize(abs(data),ts_id,lab_x,lab_y,tx,tx2,dr_ts,dr_hm,times,frequencies,save_fig,dpi)

    # Z-SCORE NORMING
        tx = 'Z-Normed data'
        normed_sp_m = self.norming(data,et.norming_type.SCALING)
        self.visualize(abs(normed_sp_m),ts_id,lab_x,lab_y,tx,tx2,dr_ts,dr_hm,times,frequencies,save_fig,dpi)

    # TRESHOLDING (NEGATIVE)
        tx = 'Thresholded (by 0) data '
        filt_normed_sp_m = self.tresholding(data=normed_sp_m,filtering_type=et.dani_filtering_type.NEGATIVE)
        self.visualize(abs(filt_normed_sp_m),ts_id,lab_x,lab_y,tx,tx2, dr_ts, dr_hm,times,frequencies,save_fig,dpi)

    #SMOOTHING (LIMITED FFT)
        tx = f"Smoothed data with limited number of FFT ({n_of_fft}) components"
        smoothed_sp_m = self.limited_fft(data=filt_normed_sp_m, num_of_fft_components=n_of_fft)
        self.visualize(abs(smoothed_sp_m), ts_id, lab_x, lab_y, tx, tx2, dr_ts, dr_hm,times,frequencies,save_fig,dpi)

    # TRESHOLDING 2
        tx = 'Thresholded (by Average) data'
        data = self.tresholding(data=smoothed_sp_m, filtering_type=et.dani_filtering_type.MEAN)
        self.visualize(abs(data), ts_id, lab_x, lab_y, tx, tx2, dr_ts, dr_hm,times,frequencies,save_fig,dpi)
        return data






    #
    #
    # def fit_original(self,data:np.ndarray, hop_length:int ,n_features:int, sr:int = 8000, n_fft:int = 10, freq_id:int = 130,
    #         dr_ts:bool=False, dr_hm:bool=False, num_of_fft_components:float=0.5, save_fig:bool=False,
    #         dpi:int=300, feature_type:fet.Feature_type = fet.Feature_type.STFT):
    #
    #     len_dim_1 = n_features
    #     len_dim_2 = int(len(data)/hop_length+1)
    #
    #     frequencies = np.arange(0, len_dim_1) * sr / ((n_features - 1) * 2)
    #     times = np.linspace(0,len(data)/sr, len_dim_2+1)
    #
    #     if self.separation_logic == et.separation_type.BY_TIME_SLICES:
    #         lab_x = "Frequency (Hz)"
    #         lab_y = "Magnitude"
    #         ts_id = min(freq_id, len_dim_1 - 1)
    #         tx2 = f" at {round(times[ts_id],2)} min"
    #         n_of_fft = int(num_of_fft_components * len_dim_1)
    #     else:
    #         lab_x = "Time"
    #         lab_y = "Amplitude"
    #         ts_id = min(freq_id, len_dim_2 - 1)
    #         tx2 = f" at {int(frequencies[ts_id])} Hz"
    #         n_of_fft = int(num_of_fft_components * len_dim_2)
    #
    # # ORIGINAL DATSET
    #     tx = 'Original data'
    #     # raw_sp_m = np.abs(librosa.stft(y=data, n_fft=(n_features - 1) * 2, hop_length=hop_length, window='hann'))
    #
    #     if feature_type == fet.Feature_type.STFT:
    #         raw_sp_m = librosa.stft(y=data, n_fft=(n_features - 1) * 2, hop_length=hop_length, window='hann')
    #     elif feature_type == fet.Feature_type.MFCC_DANI:
    #         raw_sp_m = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_features, hop_length=hop_length, n_fft=n_fft)
    #     elif feature_type == fet.Feature_type.MFCC_DELTA_DANI:
    #         mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_features, hop_length=hop_length, n_fft = n_fft)
    #         raw_sp_m = librosa.feature.delta(data = mfcc, width=3, order=1)
    #     elif feature_type == fet.Feature_type.CHROMA_DANI:
    #         raw_sp_m = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=n_features, hop_length=hop_length)
    #
    #
    #
    #     #self.visualize(abs(raw_sp_m),ts_id,lab_x,lab_y,tx,tx2,dr_ts,dr_hm,times,frequencies,save_fig,dpi)
    #
    #
    #
    #
    #
    # # Z-SCORE NORMING
    #     tx = 'Z-Normed data'
    #     normed_sp_m = self.norming(raw_sp_m,et.norming_type.SCALING)
    #     #self.visualize(abs(normed_sp_m),ts_id,lab_x,lab_y,tx,tx2,dr_ts,dr_hm,times,frequencies,save_fig,dpi)
    #
    # # TRESHOLDING (NEGATIVE)
    #     tx = 'Thresholded (by 0) data '
    #     filt_normed_sp_m = self.tresholding(data=normed_sp_m,filtering_type=et.filtering_type.NEGATIVE)
    #     #self.visualize(abs(filt_normed_sp_m),ts_id,lab_x,lab_y,tx,tx2, dr_ts, dr_hm,times,frequencies,save_fig,dpi)
    #
    # #SMOOTHING (LIMITED FFT)
    #     tx = f"Smoothed data with limited number of FFT ({n_of_fft}) components"
    #     smoothed_sp_m = self.limited_fft(data=filt_normed_sp_m, num_of_fft_components=n_of_fft)
    #     #self.visualize(abs(smoothed_sp_m), ts_id, lab_x, lab_y, tx, tx2, dr_ts, dr_hm,times,frequencies,save_fig,dpi)
    #
    # # TRESHOLDING 2
    #     tx = 'Thresholded (by Average) data'
    #     data = self.tresholding(data=smoothed_sp_m, filtering_type=et.filtering_type.MEAN)
    #     #self.visualize(abs(data), ts_id, lab_x, lab_y, tx, tx2, dr_ts, dr_hm,times,frequencies,save_fig,dpi)
    #     return data
