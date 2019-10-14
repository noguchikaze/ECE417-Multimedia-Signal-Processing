import unittest,os,wave,json,math
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight
from parameterized import parameterized
from submitted import Spectrograph
import score

# Define the processing steps
steps = [
    'nframes',
    'frames',
    'hammingwindow',
    'wframes',
    'timeaxis',
    'dftlength',
    'zvalues',
    'stft',
    'levels',
    'image'
]

# TestSequence
class TestSequence(unittest.TestCase):
    @parameterized.expand([
        ['file%d_%s'%(testcase,steps[step]),testcase,step]
        #for testcase in range(0,5)
        for testcase in range(0, 1)
        for step in range(0,10)
        ])
    @weight(2)
    def test_sequence(self, name, testcase, step):
        # Read the audio
        data_filename = 'data/file%d.wav'%(testcase)
        w = wave.open(data_filename,'rb') 
        samplerate = w.getframerate()
        nsamples = w.getnframes()
        signal = np.frombuffer(w.readframes(nsamples),dtype=np.int16).astype('float32')/32768
        w.close()

        # Even test cases are wideband (6ms frame), odd test cases are narrowband (25ms frame)
        framelength = int(math.ceil(samplerate*0.006)) if testcase%2==0 else int(math.ceil(samplerate*0.025))
        frameskip = int(round(samplerate*0.002))
        self.graph = Spectrograph(signal,samplerate,framelength,frameskip,480,6000,60)

        # Load the reference solutions
        criteria_filename = 'solutions/file%d_criteria.json'%(testcase)
        with open(criteria_filename) as f:
            ref = json.load(f)            

        # Run all of the steps up through the current step
        # This wastes a huge amount of compute time, but it's necessary for independent unit tests
        for n in range(0,step+1):
            getattr(self.graph, 'set_' + steps[n])()

        # Get the attribute being tested by this particular unit test
        x = getattr(self.graph, steps[int(step)])
        # Check that it has the correct data size
        self.assertEqual(score.hash_data_size(x), ref[steps[step]]['size'])
        # Check that its golden-mean-projections all have the correct values
        self.assertEqual(score.score_data_content(x,ref[steps[step]]['content'],testcase,step,3,0.001),3)

