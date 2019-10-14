import numpy as np
import wave,math

steps = [
    'frames',
    'autocor',
    'lpc',
    'stable',
    'pitch',
    'logrms',
    'logsigma',
    'samplepitch',
    'excitation',
    'synthesis'
]
    

class Dataset(object):
    """
    dataset=Dataset(testcase): load the waveform for the specified testcase
    Result: 
    dataset.signal is the waveform, as a numpy array
    dataset.samplerate is the sampling rate
    dataset.framelength is set to 30ms always
    dataset.frameskip is set to half of framelength always
    dataset.nframes is set to the right number of frames
    dataset.order is set to 12 always, the useful order of the LPC
    """
    def __init__(self,testcase):
        w = wave.open('data/file%d.wav'%(testcase),'rb') 
        self.samplerate = w.getframerate()
        self.signal = np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype('float32')/32768
        w.close()
        self.framelength = round(0.03*self.samplerate)
        self.frameskip = round(0.015*self.samplerate)
        self.nframes = 1+int(math.ceil((len(self.signal)-self.framelength)/self.frameskip))
        self.order = 12
        
   # PROBLEM 4.0
    #
    # Chop the waveform into frames
    # self.frames[t,n] should be self.signal[t*self.frameskip+n]
    def set_frames(self):
        self.frames = np.zeros((self.nframes,self.framelength))
        # TODO: fill the frames
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
        
    # PROBLEM 4.1
    #
    # Find the autocorrelation function of each frame
    # self.autocor[t,self.framelength+m-1] should equal R[m],
    #   where R[m] = sum_n frame[n] frame[n+m].
    def set_autocor(self):
        self.autocor = np.zeros((self.nframes,2*self.framelength-1))
        # TODO: compute autocor for each frame
        #R = np.zeros(2*self.framelength-1)
        for t in range( self.nframes ): 
            x = self.frames[t]
            self.autocor[t,:] = np.convolve( x, x[::-1], 'full')

    # PROBLEM 4.2
    #
    # Calculate the LPC coefficients in each frame
    # lpc = inv(R)*gamma, where R and gamma are the autocor matrix and vector, respectively
    def set_lpc(self):
        self.lpc = np.zeros((self.nframes,self.order))
        # TODO: for each frame, compute R, compute gamma, compute lpc
        AC = np.zeros((self.order, self.order))   # 12x12 matrix
        for n in range(self.nframes):
            for i in range(self.order):
                for j in range(self.order):
                    m = abs(i-j)
                    AC[i,j] = self.autocor[n, self.framelength-1+m]
            Rinv = np.linalg.inv(AC)
            self.lpc[n] = np.dot( Rinv , self.autocor[n,self.framelength:self.framelength+self.order])          
    # PROBLEM 4.3
    #
    # Create the inverse of a stable synthesis filter.
    #   First, find the LPC inverse filter polynomial: [1, -a[0], -a[1], ..., -a[order-1]].
    #   Second, find its roots, using np.roots.
    #   Third, truncate magnitude of the roots:
    #     if any root, r, has absolute(r)>0.999, then replace it with 0.999*np.exp(1j*np.angle(r)).
    #   Finally, reconstruct a stable inverse filter (of length order+1) using np.poly(r).
    def set_stable(self):
        self.stable = np.zeros((self.nframes,self.order+1))
        # TODO: (1) create the inverse filter, (2) find its roots, (3) truncate magnitude, (4) find poly
        # create inv filter
        poly = np.zeros(self.order+1)
        poly[0] = 1
        for n in range(self.nframes):
            for i in range(1,self.order+1):
                poly[i] = -self.lpc[n, i-1]
            # find the root
            root = np.roots(poly)
            # truncate magnitude
            for r in range(len(root)):
                if abs(root[r])>0.999:
                    root[r] = 0.999*np.exp(1j*np.angle(root[r]))
            # reconstruct the polynomial
            self.stable[n] = np.poly1d(root, r=True)

    # PROBLEM 4.4
    #
    # Calculate the pitch period in each frame:
    #   self.pitch[t] = 0 if the frame is unvoiced
    #   self.pitch[t] = pitch period, in samples, if the frame is voiced.
    #   Pitch period should maximize R[pitch]/R[0], in the range ceil(0.004*Fs) <= pitch < floor(0.013*Fs)
    #   Call the frame voiced if and only if R[pitch]/R[0] >= 0.25.
    def set_pitch(self):
        self.pitch = np.zeros(self.nframes)
        # TODO: for each frame, find maximum normalized autocor in the range between minpitch and maxpitch
        offset = np.argmax(self.autocor[0])
        Pmax = offset + int(np.floor(0.013*self.samplerate))
        Pmin = offset + int(np.ceil(0.004*self.samplerate))
        for n in range(self.nframes):
            # L = shift for a period
            P = int(Pmin + np.argmax(self.autocor[n,Pmin:Pmax]))
            if (self.autocor[n,offset] != 0) and ((self.autocor[n,P]/self.autocor[n,offset]) >= 0.25):
                self.pitch[n] = P-offset
            else:
                pass
        
    # PROBLEM 4.5
    #
    # Calculate the log(RMS) each frame
    # RMS[t] = root(mean(square(frame[t,:])))
    def set_logrms(self):
        self.logrms = np.zeros((self.nframes))
        # TODO: calculate log RMS of samples in each frame
        for t in range(self.nframes):
            s = 0
            for i in range(self.framelength):
                s += (self.frames[t, i])**2
            s = s/self.framelength
            self.logrms[t] = np.log(np.sqrt(s))
        # print(self.logrms)

    # PROBLEM 4.6
    #
    # Linearly interpolate logRMS, between frame boundaries,
    #  in order to find the log standard deviation of each output sample.
    #  logsigma[t,n] = logrms[t]*(frameskip-n)/frameskip + logrms[t+1]*n/frameskip
    def set_logsigma(self):
        self.logsigma = np.zeros((self.nframes-1,self.frameskip))
        # TODO: linearly interpolate logRMS between frame boundaries
        for t in range(self.nframes-1):
            for n in range(self.frameskip):
                self.logsigma[t,n] = self.logrms[t]*(self.frameskip-n)/self.frameskip + self.logrms[t+1]*n/self.frameskip

    # PROBLEM 4.7
    #
    # Linearly interpolate pitch, between frame boundaries,
    # If t and t+1 voiced: samplepitch[t,n] = pitch[t]*(frameskip-n)/frameskip + pitch[t+1]*n/frameskip
    # If only t voiced: samplepitch[t,n] = pitch[t]
    # If t unvoiced: samplepitch[t,n] = 0
    def set_samplepitch(self):
        self.samplepitch = np.zeros((self.nframes-1,self.frameskip))
        # TODO: linearly interpolate pitch between frame boundaries (if both nonzero)
        for t in range(self.nframes-1):
            if self.pitch[t]*self.pitch[t+1] != 0:  # both voiced
                for n in range(self.frameskip):
                    self.samplepitch[t,n] = self.pitch[t]*(self.frameskip-n)/self.frameskip + self.pitch[t+1]*n/self.frameskip
            elif self.pitch[t]>0:   # only t voiced
                self.samplepitch[t, :] = self.pitch[t]
            else:   # only t+1 voiced
                self.samplepitch[t,:] = 0

                
    # PROBLEM 4.8
    #
    # Synthesize the output excitation signal
    # unvoiced: self.excitation[t,:]=np.random.normal
    #    WARNING: the call to np.random.seed(0), below, makes your "random" numbers the same as
    #    the "random" numbers in the solution, if you generate the same number of them.
    #    Keep that line there -- it's the only way to get your code to pass the autograder.
    # voiced: you need to keep a running tally of the pitch phase, from voiced frame to voiced frame.
    #    phase increments, at every sample, by 2pi/samplepitch[n].
    #    Whenever the phase passes 2pi, there is a pitch pulse.
    #    The magnitude of the pitch pulse is sqrt(samplepitch), to make sure RMS=1.
    def set_excitation(self):
        self.excitation = np.zeros((self.nframes-1,self.frameskip))
        np.random.seed(0)
        pi = np.pi
        phi = 0
        # TODO: create white noise for unvoiced frames, pulse train for voiced frames
        for t in range(self.nframes-1):
            if self.pitch[t] == 0:  # for the unvoiced
                self.excitation[t,:] = np.random.normal(size = (1,self.frameskip))               
                #for n in range(self.frameskip):
                #    phi += 2*pi/self.samplepitch[t,n]
            else:   # for the voiced
                for n in range(self.frameskip):
                    phi += 2*pi/self.samplepitch[t,n]
                    if phi>=2*pi:
                        phi -= 2*pi
                        self.excitation[t,n] = np.sqrt(self.samplepitch[t,n])
        

    # PROBLEM 4.9
    #
    # Synthesize the speech.
    #    x = np.reshape(np.exp(self.logsigma)*self.excitation,-1)
    #    synthesis[n] = x[n] - sum_{m=1}^{order}(stable[t,m]*synthesis[n-m])
    #    where t=int(np.floor(n/frameskip))
    #    and you can assume that synthesis[n-m]=0 for n-m < 0.
    def set_synthesis(self):
        self.synthesis = np.zeros((self.nframes-1)*self.frameskip)
        # TODO: fill the filter buffer, then reshape it to create self.synthesis
        x = np.reshape(np.exp(self.logsigma)*self.excitation,-1)
        for n in range(len(self.synthesis)):
            Sum = 0
            for m in range(1,self.order+1):
                if (n-m)<0:
                    pass
                else:
                    t = int(np.floor(n/self.frameskip))
                    Sum += self.stable[t,m]*self.synthesis[n-m]
            self.synthesis[n] = x[n] - Sum
