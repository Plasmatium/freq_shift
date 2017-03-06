# https://blog.blahgeek.com/yong-pythonba-zhou-jie-lun-bian-wei-zhou-jie-lun.html

import numpy as np
import wave
from pylab import *
from IPython.core.debugger import Tracer
from functools import reduce
set_trace = Tracer()

class FFTShift:
    def __init__(self, fn):
        self.fn = fn
        self.wav = wave.open(fn)
        self.datatype = np.dtype('<i'+str(self.wav.getsampwidth()))
        self.raw_data = np.fromstring(self.wav.readframes(self.wav.getnframes()), self.datatype)
        #self.fft_data = np.fft.rfft(self.raw_data)
        self.fft_data = None

    def to_wav(self):
        if self.fft_data is not None:
            return np.fft.irfft(self.fft_data)

    def to_file(self, fn):
        file_data = np.array(self.buffer_data, dtype=self.datatype).tostring()
        new_file = wave.open(fn, 'w')
        new_file.setparams(self.wav.getparams())
        new_file.writeframes(file_data)
        new_file.close()

    def tune(self, k):
        wav_set = wav_split(self.raw_data)
        new_set = [shift(data, k) for data in wav_set]
        flat_data = []
        for i in range(len(new_set)):
            flat_data.append(new_set[i])
        self.buffer_data = np.hstack(flat_data)



def shift(data, k):
    fft_data = np.fft.rfft(data)
    new_fft = [fft_data[round(i/k)] for i in range(len(fft_data))]
    return np.fft.irfft(new_fft)

def wav_split(raw, size=1024):
    chunk_num = int(len(raw)/size)
    new = [raw[size*seg : size*(seg+1)] for seg in range(chunk_num)]
    new.append(raw[size*chunk_num:])
    return new
