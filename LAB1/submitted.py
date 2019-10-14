import numpy as np
import cmath,math

class Spectrograph(object):
    """Spectrograph: a device that computes a spectrogram."""
    def __init__(self,signal,samplerate,framelength,frameskip,numfreqs,maxfreq,dbrange):
        self.signal = signal   # A numpy array containing the samples of the speech signal
        self.samplerate = samplerate # Sampling rate [samples/second]
        self.framelength = framelength # Frame length [samples]
        self.frameskip = frameskip # Frame skip [samples]
        self.numfreqs = numfreqs # Number of frequency bins that you want in the spectrogram
        self.maxfreq = maxfreq # Maximum frequency to be shown, in Hertz
        self.dbrange = dbrange # All pixels that are dbrange below the maximum will be set to zero

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
        self.nframes = 1 + int(np.ceil((N-L)/k))
       
        # TODO: set self.nframes to something else

    # PROBLEM 1.2
    # Chop the signal into overlapping frames
    # Result: self.frames is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_frames(self):
        self.frames = np.zeros((self.nframes,self.framelength),dtype='float64')
        #frame 是一個縱向有nframes個 橫向有framelength 個的array
        #先用常數函數把frame切出來
        N = len(self.signal)  # number of samples of the signal
        L = self.framelength
        k = self.frameskip
        for i in range(self.nframes):  # 總共有nframes次的i => 0-nf
            offset = i*self.frameskip  # 信號開始切的點

            if (N-L) % k == 0:
                self.frames[i] = self.signal[offset: offset + self.framelength]
            else:  # need zero padding for the last frame
                if i == self.nframes-1:  # last frame
                    self.frames[i] = np.hstack(
                        (self.signal[offset: N], np.zeros(L-(N-offset), dtype=int)))
                else:
                    self.frames[i] = self.signal[offset: offset + self.framelength]
        # TODO: fill self.frames

    # PROBLEM 1.3
    #
    # Window each frame with a Hamming window of the same length (use np.hamming)
    # Result: self.hammingwindow is a numpy.ndarray, shape=(framelength), dtype='float64'
    def set_hammingwindow(self):
        self.hammingwindow = np.zeros(self.framelength, dtype='float64')
        w = np.hamming(self.framelength)
        self.hammingwindow = w
        # TODO: fill self.hammingwindow

    # PROBLEM 1.4
    #
    # Window each frame with a Hamming window of the same length (use np.hamming)
    # Result: self.wframes is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_wframes(self):
        self.wframes = np.zeros(self.frames.shape, dtype='float64')
        for i in range(0, self.nframes):  # 如果index沒錯的話可以這樣乘XD
            self.wframes[i] = np.multiply(self.frames[i], self.hammingwindow)
        # TODO: fill self.wframes

    # PROBLEM 1.5
    #
    # Time alignment, in seconds, of the first sample of each frame, where signal[0] is at t=0.0
    # Result: self.timeaxis is a numpy.ndarray, shape=(nframes), dtype='float32'
    def set_timeaxis(self):
        self.timeaxis = np.zeros(self.nframes, dtype='float32')
        #一個sample是(1/fs)秒 因此x[0] = 0.0 x[1] = (1/fs) x[n] = (n/fs)
        #但因為有frameskip 所以是index*frameskip/fs
        fs = self.samplerate
        for i in range(0,self.nframes):
            self.timeaxis[i] = i*self.frameskip/fs
        # TODO: fill self.timeaxis    

    # PROBLEM 1.6
    #   Length of the desired DFT.
    #   You want this to be long enough so that, in numfreqs bins, you get exactly maxfreq Hertz.
    #   result: self.dftlength is an integer
    def set_dftlength(self):
        self.dftlength = 1024 # Not the correct value
        #resolution of DFT = fs/N[Hz] 反過來就是一次框起 N/fs[sec]的信號
        #在stft的過程中用wideband 框起nfreq個bin
        #bin = N*maxfreq/fs ?????
        self.dftlength = int(np.ceil(self.samplerate*self.numfreqs/self.maxfreq))
        # TODO: set self.dftlength

    # PROBLEM 1.7
    #
    # Compute the Z values (Z=exp(-2*pi*k*n*j/dftlength) that you will use in each DFT of the STFT.
    #    result (numpy array, shape=(numfreqs,framelength), dtype='complex128')
    #    result: self.zvalues[k,n] = exp(-2*pi*k*n*j/self.dftlength)
    def set_zvalues(self):
        self.zvalues = np.zeros((self.numfreqs,self.framelength), dtype='complex128')
        for k in range(0,self.numfreqs):
            for n in range(0,self.framelength):
                self.zvalues[k, n] = cmath.exp((-2*math.pi*k*n/self.dftlength)*1j)
        # TODO: fill self.zvalues

    # PROBLEM 1.8
    #
    # Short-time Fourier transform of the signal.
    #    result: self.stft is a numpy array, shape=(nframes,numfreqs), dtype='complex128'
    #    self.stft[m,k] = sum(wframes[m,:] * zvalues[k,:])
    def set_stft(self):
        self.stft = np.zeros((self.nframes,self.numfreqs), dtype='complex128')
        for m in range(0,self.nframes):
            for k in range(0,self.numfreqs):
                #mul = np.multiply(self.wframes[m, :], self.zvalues[k, :])
                #self.stft[m,k] = np.sum(mul)
                self.stft[m,k] = np.sum(np.multiply(self.wframes[m,:],self.zvalues[k,:]))
                #self.maxval = max(abs(self.stft[0][0]),abs(self.stft[m][k]))
        # TODO: fill self.stft

    # PROBLEM 1.9
    #
    # Find the level (in decibels) of the STFT in each bin.
    #    Normalize so that the maximum level is 0dB.
    #    Cut off small values, so that the lowest level is truncated to -60dB.
    #    result: self.levels is a numpy array, shape=(nframes,numfreqs), dtype='float64'
    #    self.levels[m,k] = max(-dbrange, 20*log10(abs(stft[m,k])/maxval))
    def set_levels(self):
        self.levels = np.zeros((self.nframes, self.numfreqs), dtype='float64')
        '''
        power = np.zeros((self.nframes, self.numfreqs), dtype='float64')
        for i in range(0, self.nframes):
            power[i] = [abs(x) for x in self.stft[i]]
        compare = np.zeros(self.nframes, dtype='float64')
        for i in range(0, self.nframes):
            compare[i] = np.max(power[i])
        maxval = np.max(compare)

        for m in range(0, self.nframes):
            for k in range(0, self.numfreqs):
                self.levels[m, k] = 20*math.log10(power[m, k]/maxval)
               # self.levels[m,k] = np.max(-self.dbrange,20*math.log10(power[m,k]/maxval)) #wrong
                if self.levels[m, k] < -self.dbrange:
                    self.levels[m, k] = -self.dbrange
        '''
        self.maxval = max(max(x) for x in abs(self.stft))   #np.amax(self.stft)
        for m in range(0,self.nframes):
            for k in range(0,self.numfreqs):
                #what is MAXVAL???
                self.levels[m, k] = max(-self.dbrange, 20*math.log((abs(self.stft[m, k])/self.maxval),10))
        # TODO: fill self.levels

    # PROBLEM 1.10
    #
    # Convert the level-spectrogram into a spectrogram image:
    #    Add 60dB (so the range is from 0 to 60dB), scale by 255/60 (so the max value is 255),
    #    and convert to data type uint8.
    #    result: self.image is a numpy array, shape=(nframes,numfreqs), dtype='uint8'
    
    def set_image(self):
        self.image = np.zeros((self.nframes,self.numfreqs), dtype='uint8')
        newlevels = np.zeros((self.nframes, self.numfreqs), dtype='float64')
        for i in range(0, self.nframes):
            newlevels[i] = [(x+60)*255/60 for x in self.levels[i]]
        self.image = newlevels.astype(np.uint8)

        
        # TODO: fill self.image
    

