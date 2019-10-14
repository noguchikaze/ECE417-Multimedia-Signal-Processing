import unittest,json,submitted,score
from gradescope_utils.autograder_utils.decorators import weight
from parameterized import parameterized

# Define the processing steps
steps = [
    'vectors',
    'mean',
    'centered',
    'transform',
    'features',
    'energyspectrum',
    'neighbors',
    'hypotheses',
    'confusion',
    'metrics',
]

# Testsequence
class TestSequence(unittest.TestCase):
    @parameterized.expand([
        ['%s%d_%dnn_step%d'%(transformtype,nfeats,K,step),nfeats,transformtype,K,step]
        for nfeats in [9,36]
        for transformtype in ['dct','pca']
        for K in [1,4]
        for step in range(0,10)
    ])
    @weight(1.25)
    def test_sequence(self, name, nfeats, transformtype, K, stepnum):
        # Create the KNN object, and load its data
        knn = submitted.KNN('data',nfeats,transformtype,K)

        # Load the reference solutions
        filename = '%s%d_%dnn'%(transformtype,nfeats,K)
        with open('solutions/%s.json'%(filename)) as f:
            ref = json.load(f)            

        # Run all of the steps up through the current step
        # This wastes a huge amount of compute time, but it's necessary for independent unit tests
        for step in steps[0:(stepnum+1)]:
            getattr(knn, 'set_' + step)()

        # Get the attribute being tested by this particular unit test
        step = steps[stepnum]
        x = getattr(knn, step)
        # Check that it has one of the valid data sizes
        hyp = score.hash_data_size(x)
        if type(ref[step]['size'])==list:
            self.assertTrue(any([ r==hyp for r in ref[step]['size'] ]))
        else: 
            self.assertEqual(hyp, ref[step]['size'])
        # Check that its golden-mean-projections all have the correct values
        self.assertTrue(score.validate_data_content(x,ref[step]['content'],name,stepnum,0.001))
