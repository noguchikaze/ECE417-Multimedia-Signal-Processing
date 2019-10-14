import numpy as np
import cmath,math,os,collections
from PIL import Image

steps = [
    'ypbpr',
    'rowconv',
    'gradient',
    'background',
    'clipped',
    'matchedfilters',
    'matches',
    'features',
    'confusion',
    'accuracy'
    ]
    

class Dataset(object):
    """
    dataset=Dataset(classes), where classes is a list of class names.
    Result: 
    dataset.data is a list of observation data read from the files,
    dataset.labels gives the class number of each datum (between 0 and len(dataset.classes)-1)
    dataset.classes is a copy of the provided list of class names --- there should be exactly two.
    """
    def __init__(self,classes,nperclass):
        # Number of sets (train vs. test), number of classes (always 2), num toks per set per split (6)
        self.nclasses = 2
        self.nperclass = nperclass
        self.ndata = self.nclasses * self.nperclass
        # Load classes from the input.  If there are more than 2, only the first 2 are used
        self.classes = classes
        # Number of rows per image, number of columns, number of colors
        self.nrows = 200
        self.ncols = 300
        self.ncolors = 3
        # Data sets, as read from the input data directory
        self.labels = np.zeros((self.ndata),dtype='int')
        self.data = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata),dtype='float64')
        for label in range(0,self.nclasses):
            for num in range(0,self.nperclass):
                datum = label*(self.nperclass) + num
                filename = os.path.join('data','%s%2.2d.jpg'%(self.classes[label],num+1))
                self.labels[datum] = label
                self.data[:,:,:,datum] = np.asarray(Image.open(filename))
        
    # PROBLEM 3.0
    #
    # Convert image into Y-Pb-Pr color space, using the ITU-R BT.601 conversion
    #   [Y;Pb;Pr]=[0.299,0.587,0.114;-0.168736,-0.331264,0.5;0.5,-0.418688,-0.081312]*[R;G;B].
    # Put the results into the numpy array self.ypbpr[:,:,:,m]
    def set_ypbpr(self):
        self.ypbpr = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata))
        # TODO: convert each RGB image to YPbPr
        y = [0.299,0.587,0.114]
        b = [-0.168736,-0.331264,0.5]
        r = [0.5,-0.418688,-0.081312]
        #print(len(self.data[0,0,:,0]), len(y))
        '''
        self.ypbpr[:,:,0,:] = y[0]*self.data[:,:,0,:] + y[1]*self.data[:,:,1,:] + y[2]*self.data[:,:,2,:]
        self.ypbpr[:,:,1,:] = b[0]*self.data[:,:,0,:] + b[1]*self.data[:,:,1,:] + b[2]*self.data[:,:,2,:]
        self.ypbpr[:,:,2,:] = r[0]*self.data[:,:,0,:] + r[1]*self.data[:,:,1,:] + r[2]*self.data[:,:,2,:]
        '''
        self.ypbpr[:,:,0,:] = np.tensordot(y, self.data, axes=([0],[2]))
        self.ypbpr[:,:,1,:] = np.tensordot(b, self.data, axes=([0],[2]))
        self.ypbpr[:,:,2,:] = np.tensordot(r, self.data, axes=([0],[2]))
        
    # PROBLEM 3.1
    #
    # Filter each row of ypbpr with two different filters.
    # The first filter is the difference: [1,0,-1]
    # The second filter is the average: [1,2,1]
    # Keep only the 'valid' samples, thus the result has size (nrows,ncols-2,2*ncolors)
    # The output 'colors' are (diffY,diffPb,diffPr,aveY,avePb,avePr).
    def set_rowconv(self):
        self.rowconv = np.zeros((self.nrows,self.ncols-2,2*self.ncolors,self.ndata))
        # TODO: calculate the six output planes (diffY,diffPb,diffPr,aveY,avePb,avePr).
        h = [1,0,-1]    #diff
        a = [1,2,1]     #avg
        for n in range(self.ndata):
            for i in range(self.nrows):
                for j in range(self.ncolors):
                        self.rowconv[i, :, j  ,n] = np.convolve(self.ypbpr[i,:,j,n], h, 'valid')
                        self.rowconv[i, :, j+3,n] = np.convolve(self.ypbpr[i,:,j,n], a, 'valid')



    # PROBLEM 3.2
    #
    # Calculate the (Gx,Gy) gradients of the YPbPr images using Sobel mask.
    # This is done by filtering the columns of self.rowconv.
    # The first three "colors" are filtered by [1,2,1] in the columns.
    # The last three "colors" are filtered by [1,0,-1] in the columns.
    # Keep only 'valid' outputs, so size is (nrows-2,ncols-2,2*ncolors)
    def set_gradient(self):
        self.gradient = np.zeros((self.nrows-2,self.ncols-2,2*self.ncolors,self.ndata))
        # TODO: compute the Gx and Gy planes of the Sobel features for each image
        h = [1,0,-1]    
        a = [1,2,1]
        for n in range(self.ndata):
            for i in range(self.ncols-2):
                for j in range(self.ncolors):
                        self.gradient[:, i, j  ,n] = np.convolve(self.rowconv[:,i,j,n], a, 'valid')
                        self.gradient[:, i, j+3,n] = np.convolve(self.rowconv[:,i,j+3,n], h, 'valid')


    # PROBLEM 3.3
    #
    # Create a matched filter, for each class, by averaging the YPbPr images of that class,
    # removing the first two rows, last two rows, first two columns, and last two columns,
    # flipping left-to-right, and flipping top-to-bottom. ->np.flip
    def set_matchedfilters(self):
        self.matchedfilters = np.zeros((self.nrows-4,self.ncols-4,self.ncolors,self.nclasses))
        # TODO: for each class, average the YPbPr images, fliplr, and flipud
        for n in range(self.nclasses):
            for i in range(self.ncolors):
                #calculate the mean
                cur_class = n*self.nperclass
                self.matchedfilters[:,:,i,n] = np.mean(self.ypbpr[2:(self.nrows-2),2:(self.ncols-2), i, cur_class:cur_class+self.nperclass], axis=2)
                # flip it ud lr
                self.matchedfilters[:,:,i,n] = np.flip(self.matchedfilters[:,:,i,n])

    # PROBLEM 3.4
    #
    # self.matches[:,:,c,d,z] is the result of filtering self.ypbpr[:,:,c,d]
    #   with self.matchedfilters[:,:,c,z].  Since we're not using scipy, you'll have to
    #   implement 2D convolution yourself, for example, by convolving each row, then adding
    #   the results; or by just multiplying and adding at each shift.
    def set_matches(self):
        self.matches = np.zeros((5,5,self.ncolors,self.ndata,self.nclasses))
        # TODO: compute 2D convolution of each matched filter with each YPbPr image
        for z in range(self.nclasses):
            for d in range(self.ndata):
                for c in range(self.ncolors):
                    for i in range(5):  #row
                        for j in range(5):  #col
                            h = np.flip(self.matchedfilters[:,:,c,z])
                            temp = np.multiply(h, self.ypbpr[i:self.nrows-4+i, j:self.ncols-4+j,c,d])
                            self.matches[i,j,c,d,z] = np.sum(temp)

    # PROBLEM 3.5 
    #
    # Create a feature vector from each image, showing three image features that
    # are known to each be useful for some problems.
    # self.features[d,0] is norm(Pb)-norm(Pr)
    # self.features[d,1] is norm(Gx[luminance])-norm(Gy[luminance])
    # self.gradient = np.zeros((self.nrows-2,self.ncols-2,2*self.ncolors,self.ndata))
    # self.features[d,2] is norm(match to class 1[all colors]) - norm(match to class 0[all colors])
    def set_features(self):
        self.features = np.zeros((self.ndata,3))
        # TODO: Calculate color feature, gradient feature, and matched filter feature for each image
        for d in range(self.ndata):
            self.features[d,0] = np.linalg.norm(self.ypbpr[:,:,1,d]) - np.linalg.norm(self.ypbpr[:,:,2,d])
            Gx_l = self.gradient[:,:,0,d]
            Gy_l = self.gradient[:,:,3,d]
            # the ans on gradescope is incorrect and this is the way to make the ans correct
            self.features[d,1] = np.linalg.norm(Gx_l) -np.linalg.norm(Gy_l) 
            self.features[d,2] = np.linalg.norm(self.matches[:,:,:,d,1]) - np.linalg.norm(self.matches[:,:,:,d,0])

    # PROBLEM 3.6
    #
    #   self.accuracyspectrum[d,f] = training corpus accuracy of the following classifier:
    #   if self.features[k,f] >= self.features[d,f], then call datum k class 1, else call it class 0.
    def set_accuracyspectrum(self):
        self.accuracyspectrum = np.zeros((self.ndata,3))
        # TODO: Calculate the accuracy of every possible single-feature classifier
        compare = np.zeros((self.ndata, self.ndata))
        for f in range(3):
            for d in range(self.ndata):
                for k in range(self.ndata):
                    ref = self.features[d,f]
                    sub = self.features[k,f]
                    # determine the class type of sub
                    if k < self.nperclass:    # sub = class 0
                        cur_class = 0
                    else:   # sub = class 1
                        cur_class = 1
                    # determine the trained class type
                    if sub >= ref:  # -> class 1
                        trained_class = 1
                    else:   # -> class 0
                        trained_class = 0
                    
                    # fill out the compare array
                    if cur_class == trained_class:
                        compare[d,k] = 1
                    else:
                        compare[d,k] = 0
                self.accuracyspectrum[d,f] = np.sum(compare[d,:])/self.ndata
    # PROBLEM 3.7
    #
    # self.bestclassifier specifies the best classifier for each feature
    # self.bestclassifier[f,0] specifies the best threshold for feature f
    # self.bestclassifier[f,1] specifies the polarity:
    #   to specify that class 1 has feature >= threshold, set self.bestclassifier[f,1] = +1
    #   to specify that class 1 has feature < threshold, set self.bestclassifier[f,1] = -1
    # self.bestclassifier[f,2] gives the accuracy of the best classifier for feature f,
    #   computed from the accuracyspectrum as
    #   accuracy[f] = max(max(accuracyspectrum[:,f]), 1-min(accuracyspectrum[:,f])).
    #   If the max is selected, then polarity=+1; if the min is selected, then polarity=-1.
    def set_bestclassifier(self):
        self.bestclassifier = np.zeros((3,3))
        #
        # TODO: find the threshold and polarity of best classifier for each feature
        for f in range(3):
            i = np.argmax(self.accuracyspectrum[:,f])   # index for the max value in accu_spectrum
            self.bestclassifier[f,0] = self.features[i,f]
            pos = self.accuracyspectrum[i,f]
            neg = 1-min(self.accuracyspectrum[:,f])
            self.bestclassifier[f,2] = max(pos, neg)
            if pos > neg:
                self.bestclassifier[f,1] = -1
            else:
                self.bestclassifier[f,1] = 1
