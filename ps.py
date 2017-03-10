# src from:
# https://bitbucket.org/snippets/igor_b/qEKz5

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

if len(sys.argv)!=5:
    print("usage: %s <in_file> <out_file> <pitch_percent> <robot>\n"%(sys.argv[0]));
    sys.exit(-1)

srate, din = wavfile.read(sys.argv[1])

ws = 2048
overlap = 8

ss = ws//overlap

w = np.hanning(ws)
b = np.zeros(ws)

dout = np.zeros(din.size)

coeff=float(sys.argv[3])/100


fpb = srate/ws  # frequency per bin
exp = 2*np.pi*ss/ws # expected phase

k = np.linspace(1, ws//2+1, ws//2+1)  # ramp
lph = np.zeros(ws//2+1)  # last phase
sum_phase = np.zeros(ws//2+1)

num_bl = int(din.size/ss)

for i in range(int((din.size-ws)/ss)):
    b = (din[i*ss:i*ss+ws] / 2**16) * w

    ff = np.fft.rfft(b)
    ffs = ff.size

    # analysis
    # --------------------------------

    re = np.real(ff)
    im = np.imag(ff)

    magn = 2.0 * np.sqrt(re**2+im**2)
    phase = np.arctan2(im, re)

    # compute phase diff
    freq = phase - lph
    lph = np.copy(phase)

    # subtract expected phase diff
    freq -= k*exp

    # map to [-pi,pi]
    while len(freq[freq<-np.pi]): freq[freq<-np.pi]+=2*np.pi
    while len(freq[freq>np.pi]): freq[freq>np.pi]-=2*np.pi
    
    # get deviation from bin
    freq = overlap*freq/2.0/np.pi

    # get partial's true freq
    freq = k*fpb + freq*fpb

    # proc
    ii=0
    synfr = np.copy(magn)
    while ii<freq.size:
        idx = int(ii*coeff)
        if idx <= freq.size-1:
            magn[idx] = synfr[ii]*coeff * (np.random.random())
        ii+=1

    # add robot
    if sys.argv[4][0]=='1':
        freq = np.zeros(freq.size)


    # synthesis
    # --------------------------------

    # subtract the bin mid freq
    freq -= k*fpb

    # get bin deviation
    freq /= fpb

    # account for overlape
    freq = 2.0*np.pi*freq/overlap

    # add the overlap phase advance
    freq += k*exp

    # accumulate delta phase
    sum_phase += freq

    ff = magn*np.cos(sum_phase) + magn*np.sin(sum_phase)*1j

    b=np.fft.irfft(ff)
    dout[i*ss:i*ss+ws] += b*w

wavfile.write(sys.argv[2], srate, dout)
    