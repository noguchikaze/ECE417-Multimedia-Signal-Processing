#unit test of pb

import numpy as np
import cmath,math


class Spectrograph(object):
    """Spectrograph: a device that computes a spectrogram."""
    def __init__(self, signal, samplerate, framelength, frameskip, numfreqs, maxfreq, dbrange):
        self.signal = signal   # A numpy array containing the samples of the speech signal
        self.samplerate = samplerate  # Sampling rate [samples/second]
        self.framelength = framelength  # Frame length [samples]
        self.frameskip = frameskip  # Frame skip [samples]
        self.numfreqs = numfreqs  # Number of frequency bins that you want in the spectrogram
        self.maxfreq = maxfreq  # Maximum frequency to be shown, in Hertz
        # All pixels that are dbrange below the maximum will be set to zero
        self.dbrange = dbrange

    # PROBLEM 1.1
    #
    # Figure out how many frames there should be
    # so that every sample of the signal appears in at least one frame,
    # and so that none of the frames are zero-padded except possibly the last one.
    #
    # Result: self.nframes is an integer
    def set_nframes(self):
        self.nframes = 1024  # Not the correct value
        N = len(self.signal)  # number of samples of the signal
        L = self.framelength
        k = self.frameskip
        #frame length and shift known
        if (N-L) % k == 0:  # just the right number
            self.nframes = 1 + int((N-L)/k)
        elif (N-L) % k != 0:
            self.nframes = 2 + int((N-L)/k)

    # PROBLEM 1.2
    #
    # Chop the signal into overlapping frames
    # Result: self.frames is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_frames(self):
        self.frames = np.zeros((self.nframes, self.framelength), dtype='float64')
        #print(self.frames)
        #frame 是一個縱向有nframes個 橫向有framelength 個的array
        #先用常數函數把frame切出來
        N = len(self.signal)  # number of samples of the signal
        L = self.framelength
        k = self.frameskip
        for i in range(self.nframes):  # 總共有nframes次的i => 0-nf
            offset = i*self.frameskip   #信號開始切的點
            if (N-L) % k == 0:
                self.frames[i] = self.signal[ offset : offset + self.framelength]
                
            else:  # need zero padding for the last frame
                if i == self.nframes-1:  # last frame
                    self.frames[i] = np.hstack((self.signal[offset : N], np.zeros(L-(N-offset), dtype=int)))
                else:
                    self.frames[i] = self.signal[offset : offset + self.framelength]


A = np.zeros(100,dtype = int)
for i in range(0,100):
    A[i] = i
a = Spectrograph(A,16000,15,10,400,16000,2000)
print("the signal looks like: \n", a.signal)
a.set_nframes()
a.set_frames()
print("the frame number of a is", a.nframes)
print(a.frames)