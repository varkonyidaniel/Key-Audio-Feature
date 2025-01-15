import sys
import numpy as np
from Enum.enum_types import Filtering_Type as ft
from Enum.enum_types import separation_type as st
from Filtering import dani_filtering as df
from scipy.signal import butter, lfilter, filtfilt
from scipy.ndimage import gaussian_filter, laplace, gaussian_laplace, median_filter, uniform_filter


def get_filter(fs:int, cutoff_freq:int, order:int, filter_type:ft):
    nyq = 0.5 * fs
    if filter_type in [ft.LOWPASS, ft.HIGHPASS]:
        Cutoff = cutoff_freq[0] / nyq
    else:
        Cutoff = (cutoff_freq[0]/nyq, cutoff_freq[1]/nyq)

    b,a = butter(N=order, Wn=Cutoff, btype=filter_type.value, analog=False, output='ba')
    return b, a


def filter_signal(data:np.ndarray, cutoff_freq, fs, order, filter_type, radius):
    if filter_type in [ft.LOWPASS, ft.HIGHPASS,ft.BANDPASS]:
        b, a = get_filter(fs=fs, cutoff_freq=cutoff_freq,order=order,filter_type =filter_type)

    if filter_type == ft.LOWPASS:
        return lfilter(b, a, data)

    elif filter_type == ft.HIGHPASS:
        return filtfilt(b, a, data)

    elif filter_type == ft.BANDPASS:
        return lfilter(b, a, data)

    elif filter_type == ft.GAUSSIAN:
            return gaussian_filter(abs(data) if data.dtype == 'complex64' else data,sigma=radius, order=order)

    elif filter_type == ft.LAPLACIAN:
        return laplace(abs(data) if data.dtype == 'complex64' else data)

    elif filter_type == ft.GAUSSIAN_LAPLACE:
        return gaussian_laplace(abs(data) if data.dtype == 'complex64' else data, sigma=radius)

    elif filter_type == ft.MEAN:
        return uniform_filter(data, size=(2*int(radius)+1))

    elif filter_type == ft.MEDIAN:
        return median_filter(abs(data) if data.dtype in ['complex64', 'complex128'] else data, size= (2*int(radius)+1))

    elif filter_type == ft.NONE:
        return data

    else:
        print("not valid filtering type")
        sys.exit(3)


def Fit(data:np.ndarray, type:ft, sampling_rate:float=8000,separation_logic:int=st.BY_TIME_SLICES.value,
        cutoff:float=100.0,order:int=1, radius:float=0.5 ):

        if type == ft.DANI:
            d_f = df.dani_filtering(separation_logic)
            return d_f.fit(data=data,sr=sampling_rate)

        elif type in [ft.LOWPASS, ft.HIGHPASS, ft.BANDPASS, ft.GAUSSIAN, ft.LAPLACIAN,
                      ft.GAUSSIAN_LAPLACE, ft.MEAN, ft.MEDIAN, ft.NONE]:


            return filter_signal(data=data, cutoff_freq=cutoff, fs=sampling_rate,order=order, filter_type=type, radius=radius)

        else:
            raise print("Undefined Feature type to extract. ERROR!!")
            sys.exit(3)
