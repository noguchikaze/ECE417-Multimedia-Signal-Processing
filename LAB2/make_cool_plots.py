import os,submitted,matplotlib,matplotlib.pyplot,sys
import numpy as np

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

# plotter
class Plotter(object):
    '''Generate a series of plots that are useful for debugging your submitted code.'''
    def __init__(self,outputdirectory):
        '''Create the specified output directory, and initialize plotter to put its plots there'''
        os.makedirs(outputdirectory,exist_ok=True)
        self.figs = [ matplotlib.pyplot.figure(n) for n in range(0,10) ]
        self.outputdirectory = outputdirectory

    def make_plots(self, datadir='data',nfeats=36,transformtype='dct', K=4, numsteps=10):
        '''Create a new KNN object, run steps 0:numsteps (default  numsteps=10), and make their plots.'''
        self.output_filename=os.path.join(self.outputdirectory,'%s%d_%dnn'%(transformtype,nfeats,K))
        self.knn = submitted.KNN(datadir=datadir,nfeats=nfeats,transformtype=transformtype,K=K)
        for n in range(0,9):
            getattr(self.knn, 'set_' + steps[n])()
            plotter_method = getattr(self, 'plot_%s'%(steps[n]))
            self.figs[n].clf()
            plotter_method(self.figs[n], self.knn)                
            self.figs[n].savefig(self.output_filename + '_step%d.png'%(n))
        
    def plot_vectors(self, f, knn):
        '''Plot the grayscale version of the first image in the database, reshaped from vectors[0]'''
        a = f.add_subplot(1,1,1)
        a.imshow(knn.vectors[0,:].reshape((knn.nrows,knn.ncols)), cmap='gray')
        a.set_title('Grayscale copy of the first image in the database')

    def plot_mean(self, f, knn):
        '''Plot the average of all images in the database'''
        a = f.add_subplot(1,1,1)
        a.imshow(knn.mean.reshape((knn.nrows,knn.ncols)), cmap='gray')
        a.set_title('Grayscale plot of the average face image')

    def plot_centered(self, f, knn):
        '''Plot, in vector form, the first image, the average image, and the difference'''
        yy = [ knn.vectors[0,:], knn.mean, knn.centered[0,:] ]
        tt = ['Vectorized first image', 'Average face image','First image minus average image']
        aa = f.subplots(nrows=3,ncols=1,sharex=True)
        for (y,t,a) in zip(yy,tt,aa):
            a.plot(y)
            a.set_title(t)
        aa[2].set_xlabel('Pixel number (64 rows of 64 pixels each)')

    def plot_transform(self, f, knn):
        '''Plot the first nine basis vectors as images'''
        aa = f.subplots(nrows=3,ncols=3,sharex=True,sharey=True)
        for c in range(0,3):
            for r in range(0,3):
                k = 3*r+c if 3*r+c <= 4 else knn.nfeats-9+3*r+c
                aa[r,c].imshow(knn.transform[k,:].reshape((knn.nrows,knn.ncols)), cmap='gray')
                aa[r,c].set_title('Basis image %d'%(k))
        
    def plot_features(self, f, knn):
        '''Plot scatter plot of the dataset, colored by person number, in first two feature dimensions'''
        aa = f.subplots(nrows=2,ncols=2)
        sc="bgrcmyk"
        for c in range(0,2):
            for r in range(0,2):
                f0=4*r+2*c
                f1=4*r+2*c+1
                aa[r,c].scatter(x=knn.features[:,f0],y=knn.features[:,f1],c=[sc[p] for p in knn.labels])
                if r==0:
                    aa[r,c].set_title('%s%d vs. %s%d'%(knn.transformtype,f1,knn.transformtype,f0))
                elif r==1:
                    aa[r,c].set_xlabel('%s%d vs. %s%d'%(knn.transformtype,f1,knn.transformtype,f0))
        f.legend(handles=[matplotlib.patches.Patch(color=sc[p],label=submitted.people[p])
                          for p in range(0,max(knn.labels)+1)])

    def plot_energyspectrum(self, f, knn):
        '''Plot the energy spectrum'''
        a = f.add_subplot(1,1,1)
        a.plot(knn.energyspectrum)
        a.set_title('%s energy spectrum, %d feats (%d pixels)'%(knn.transformtype,knn.nfeats,knn.npixels))
        a.set_xlabel('Number of features')
        a.set_ylabel('Fraction of variance explained')
        
    def plot_neighbors(self, f, knn):
        '''Images showing nearest neighbor of the first and last images'''
        aa = f.subplots(nrows=knn.K+1,ncols=2,sharex=True,sharey=True)
        for c in range(0,2):
            datum = [0,knn.ndata-1][c]
            aa[0,c].imshow(knn.data[datum,:,:],cmap='gray')
            aa[0,c].set_title('Image %d'%(datum))
            for k in range(0,knn.K):
                aa[k+1,c].imshow(knn.data[knn.neighbors[datum,k],:,:],cmap='gray')
                aa[k+1,c].set_title('Image %d Neighbor %d'%(datum,k))

    def plot_hypotheses(self, f, knn):
        '''Show two misclassified images, and the nearest neighbor that caused the misclassification'''
        axis_row = -1
        datum = 0
        aa = f.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
        for datum in range(0,knn.ndata):
            if axis_row < 1 and knn.hypotheses[datum] != knn.labels[datum]:
                axis_row = axis_row + 1
                aa[axis_row,0].imshow(knn.data[datum,:,:],cmap='gray')
                aa[axis_row,0].set_title('Image %d was misclassified...'%(datum))
                dominant_neighbors=[neighbor for neighbor in knn.neighbors[datum,:]
                                    if knn.labels[neighbor]==knn.hypotheses[datum]]
                aa[axis_row,1].imshow(knn.data[dominant_neighbors[0],:,:],cmap='gray')
                aa[axis_row,1].set_title('...because it resembles image %d'%(dominant_neighbors[0]))

    def plot_confusion(self, f, knn):
        '''Show the confusion matrix as a grayscale image, with acc, recall, and precision'''
        a = f.add_subplot(1,1,1)
        a.imshow(knn.confusion)
        for r in range(0,4):
            for h in range(0,4):
                a.text(h,r,'%d'%(knn.confusion[r,h]))
        a.set_ylabel('Reference Label')
        a.set_xlabel('Hypothesis Label')
        knn.set_metrics()  # create metrics, so we can list accuracy, recall, and precision
        a.set_title('Confusion Matrix: Accuracy=%d%%, Recall=%d%%, Precision=%d%%'%(round(100*knn.metrics[0]),round(100*knn.metrics[1]),round(100*knn.metrics[2])))


#####################################################
# If this is called from the command line,
# create a plotter with the specified arguments (or with default arguments),
# then create the corresponding plots in the 'make_cool_plots_outputs' directory.
if __name__ == '__main__':
    nfeats = 36 if len(sys.argv) < 2 else int(sys.argv[1])
    transformtype = 'dct' if len(sys.argv) < 3 else sys.argv[2]
    K = 4 if len(sys.argv) < 4 else int(sys.argv[3])

    plotter=Plotter('make_cool_plots_outputs')
    plotter.make_plots(datadir='data',nfeats=nfeats,transformtype=transformtype, K=K, numsteps=10)
    
