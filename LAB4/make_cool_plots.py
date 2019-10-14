import os,submitted,matplotlib,matplotlib.pyplot,math,wave,struct
import numpy as np

# Define the processing steps
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
    
# plotter
class Plotter(object):
    '''Generate a series of plots that are useful for debugging your submitted code.'''
    def __init__(self,outputdirectory):
        '''Create the specified output directory, and initialize plotter to put its plots there'''
        os.makedirs(outputdirectory,exist_ok=True)
        self.figs = [ matplotlib.pyplot.figure(n) for n in range(0,10) ]
        self.outputdirectory = outputdirectory

    def make_plots(self, testcase):
        '''Create a new dataset object, run all the steps, and make all the plots'''
        self.output_filename=os.path.join(self.outputdirectory,'file%d'%(testcase))
        self.dataset = submitted.Dataset(testcase)
        for n in range(0,10):
            getattr(self.dataset, 'set_' + steps[n])()
            plotter_method = getattr(self, 'plot_%s'%(steps[n]))
            self.figs[n].clf()
            plotter_method(self.figs[n], self.dataset)                
            self.figs[n].savefig(self.output_filename + '_step%d.png'%(n))
        for attr in [ 'signal', 'synthesis' ]:
            w=wave.open(os.path.join(self.outputdirectory,'file%d_%s.wav'%(testcase,attr)),'w')
            w.setnchannels(1)
            w.setsampwidth(2)
            signal = getattr(self.dataset, attr)
            w.setframerate(self.dataset.samplerate)
            xmax = np.amax(np.absolute(signal))
            for x in signal:
                data = struct.pack('<h', int(32768*x/(xmax+1e-6)))
                w.writeframesraw(data)
            w.close()
            
        
    def plot_frames(self, f, dataset):
        '''Plot the frame with the largest energy'''
        bestframe=np.argmax(np.sum(np.square(dataset.frames),axis=1))
        a = f.add_subplot(1,1,1)
        a.plot(np.arange(dataset.framelength)/dataset.samplerate,dataset.frames[bestframe,:])
        a.set_title('Frame %d (%d samples at %dHz)'%(bestframe,dataset.framelength,dataset.samplerate))
        a.set_xlabel('Time (seconds)')

    def plot_autocor(self, f, dataset):
        '''Plot the autocor of the frame with the largest energy'''
        bestframe=np.argmax(np.sum(np.square(dataset.frames),axis=1))
        a = f.add_subplot(1,1,1)
        n = np.arange(1-dataset.framelength,dataset.framelength)
        a.plot(n,dataset.autocor[bestframe,:])
        a.set_title('Autocorrelation of frame %d'%(bestframe))
        a.set_xlabel('Lag (in samples) at samplerate %d'%(dataset.samplerate))

    def plot_lpc(self, f, dataset):
        '''Plot the frequency response of the lpc polynomial of the frame with highest energy'''
        bestframe=np.argmax(np.sum(np.square(dataset.frames),axis=1))
        poly = np.zeros(dataset.order+1)
        poly[0]=1
        poly[1:(dataset.order+1)]=-dataset.lpc[bestframe,:]
        inversefilter = np.zeros(101,dtype='complex128')
        for k in range(101):
            for m in range(dataset.order+1):
                inversefilter[k] += poly[m]*np.exp(-1j*math.pi*m*k/100)
        a = f.add_subplot(2,1,1)
        a.plot(poly)
        a.set_title('LPC Polynomial Coefficients, Frame %d'%(bestframe))
        b = f.add_subplot(2,1,2)        
        b.plot(np.linspace(0,dataset.samplerate/2,101),-20*np.log10(np.absolute(inversefilter)))
        b.set_title('Freq Response Level of frame %d'%(bestframe))
        b.set_xlabel('Frequency (Hertz)')
        b.set_ylabel('Level (dB)')

    def plot_stable(self, f, dataset):
        '''Plot the stabilified frequency response of the frame with the highest energy'''
        bestframe=np.argmax(np.sum(np.square(dataset.frames),axis=1))
        inversefilter = np.zeros(101,dtype='complex128')
        for k in range(101):
            for m in range(dataset.order+1):
                inversefilter[k] += dataset.stable[bestframe,m]*np.exp(-1j*math.pi*m*k/100)
        a = f.add_subplot(2,1,1)
        a.plot(dataset.stable[bestframe,:])
        a.set_title('LPC Polynomial Coefficients, Frame %d'%(bestframe))
        b = f.add_subplot(2,1,2)        
        b.plot(np.linspace(0,dataset.samplerate/2,101),-20*np.log10(np.absolute(inversefilter)))
        b.set_title('Freq Response Level of frame %d'%(bestframe))
        b.set_xlabel('Frequency (radians/sample)')
        b.set_ylabel('Level (dB)')

    def plot_pitch(self, f, dataset):
        '''Plot the pitch periods as a function of frame number'''
        a = f.add_subplot(1,1,1)
        a.plot(dataset.pitch)
        a.set_title('Pitch period as a function of frame index')
        
    def plot_logrms(self, f, dataset):
        '''Plot log RMS as a function of frame number'''
        a = f.add_subplot(1,1,1)
        a.plot(dataset.logrms)
        a.set_title('Log RMS as a function of frame index')
        
    def plot_logsigma(self, f, dataset):
        '''Plot logsigma, downsampled by a factor of 10, overlaid on waveform downsampled by 10'''
        a = f.add_subplot(1,1,1)
        a.plot(np.reshape(dataset.logsigma,-1))
        a.set_title('Log Sigma as a function of output sample number')
               
    def plot_samplepitch(self, f, dataset):
        '''Plot pitch as a function of sample index, downsampled by 10, compared to pitch period'''
        a = f.add_subplot(1,1,1)
        a.plot(np.reshape(dataset.samplepitch,-1))
        a.set_title('Output pitch period as a function of sample number')
        
    def plot_excitation(self, f, dataset):
        '''Plot excitation signal of the frame with highest amplitude'''
        bestframe=np.argmax(np.sum(np.square(dataset.frames),axis=1))
        a = f.add_subplot(1,1,1)
        a.plot(dataset.excitation[bestframe,:])
        a.set_title('Excitation of output frame %d'%(bestframe))

    def plot_synthesis(self, f, dataset):
        '''Plot highest-magnitude section of synthesized and original signals'''
        nmax = np.argmax(np.absolute(dataset.signal))
        n = np.arange(max(nmax-550,0),min(nmax+550,len(dataset.synthesis)-1))
        aa = f.subplots(2,1)
        aa[0].plot(n/dataset.samplerate,dataset.signal[n])
        aa[0].set_ylabel('Original')
        aa[1].plot(n/dataset.samplerate,dataset.synthesis[n])
        aa[1].set_ylabel('Synthetic')
        aa[1].set_xlabel('Time (seconds)')
        

#####################################################
# If this is called from the command line,
# create a plotter with the specified arguments (or with default arguments),
# then create the corresponding plots in the 'make_cool_plots_outputs' directory.
if __name__ == '__main__':
    testcase = 0

    plotter=Plotter('make_cool_plots_outputs')
    plotter.make_plots(testcase)
    
