import unittest,json,submitted,score
from gradescope_utils.autograder_utils.decorators import weight
from parameterized import parameterized

# Define the processing steps
steps = ['frames','autocor','lpc','stable','pitch','logrms','logsigma','samplepitch','excitation','synthesis']

# Testsequence
class TestSequence(unittest.TestCase):
    @parameterized.expand([
        ['file%d_step%d'%(testcase,stepnum),testcase,stepnum]
        for testcase in range(5)
        for stepnum in range(10)
    ])
    @weight(2)
    def test_sequence(self, name, testcase, stepnum):
        # Create the KNN object, and load its data
        dataset = submitted.Dataset(testcase)

        # Load the reference solutions
        filename = 'file%d'%(testcase)
        with open('solutions/%s.json'%(filename)) as f:
            ref = json.load(f)            

        # Run all of the steps up through the current step
        # This wastes a huge amount of compute time, but it's necessary for independent unit tests
        for step in steps[0:(stepnum+1)]:
            getattr(dataset, 'set_' + step)()

        # Get the attribute being tested by this particular unit test
        step = steps[stepnum]
        x = getattr(dataset, step)
        # Check that it has one of the valid data sizes
        hyp = score.hash_data_size(x)
        if type(ref[step]['size'])==list:
            self.assertTrue(any([ r==hyp for r in ref[step]['size'] ]))
        else: 
            self.assertEqual(hyp, ref[step]['size'])
        # Check that its golden-mean-projections all have the correct values
        self.assertTrue(score.validate_data_content(x,ref[step]['content'],name,stepnum,0.001))
