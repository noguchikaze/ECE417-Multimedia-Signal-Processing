import math
import numpy as np

# Convert the type or dtype to a string
def hash_data_type(x):
    if type(x)==np.ndarray:
        return('%s %s'%(str(np.ndarray),str(x.dtype)))
    else:
        return(str(type(x)))

# Convert the length or shape to a string
def hash_data_size(x):
    if type(x)==np.ndarray:
        return(str(x.shape))
    else:
        return('1')

# A convenience function to score a (possibly real or complex-valued) ndarray
#   by projecting it onto complex sequences of three different frequencies.
# To score sequences, this function multiplies by a complex exponential, with
#   a frequency that is not an integer multiple or divisor of pi.
#   Any rational number would satisfy that requirement, of course, since pi is irrational.
#   But we use powers of the golden ratio, just because it's entertaining to do so.
# Each resulting projection is scored as a string with four significant figures,
#   in an attempt to keep the score from being affected by floating-point roundoff errors.
#
goldenpowers=[(math.sqrt(5)-1)/2, 1, (math.sqrt(5)+1)/2]
def project_ndarray(x,glist):
    testseq = x.reshape(-1)
    N = len(testseq)
    p=[]
    for g in glist:
        p.append(np.inner(testseq, np.exp(1j*np.linspace(0,N,N,endpoint=False)*g)))
    return(p)
#
# Convert the output of project_ndarray to a string
def hash_data_content(x):
    if type(x)==np.ndarray:
        return(','.join([str(q) for q in project_ndarray(x,goldenpowers)]))
    else:
        return(str(x))
#
# Compare data content to a reference string, with a permitted difference specified by "tolerance"
def score_data_content(x, refstr, testcase, step, score, tolerance):
    hypstr = hash_data_content(x)
    if type(x)==np.ndarray:
        hyp = project_ndarray(x,goldenpowers)
        ref = [complex(q) for q in refstr.split(',')]
        while len(hyp) < len(ref):
            hyp.append(0)
        while len(ref) < len(hyp):
            ref.append(0)
        num = sum([abs(hyp[n]-ref[n]) for n in range(0,len(ref))])
        denom = 1e-6+sum([abs(y) for y in ref])
        if num/denom < tolerance:
            print('    CORRECT! file%d-%s-content correct w/value %s'%(testcase,step,hypstr))
            return(score)
        else:
            print('    *** error *** file%d-%s-content=%s, differs by %g%% from correct value %s'%
                  (testcase,step,hypstr,100*(num/denom),refstr))
            return(0)
            
    else:
        return(score_string(hypstr,refstr,testcase,step,'content',score))

# A convenience function to score a given test element
def score_string(hyp, ref, testcase, step, part, score):
    if hyp==ref:
        print('    CORRECT: file%d-%s-%s correct w/value %s'%(testcase, step, part, hyp))
        return(score)
    else:
        print('    *** error *** file%d-%s-%s has value %s, should be %s'%(testcase,step,part,hyp,ref))
        return(0)

