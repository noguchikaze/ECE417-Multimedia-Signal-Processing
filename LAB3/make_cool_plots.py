import os,submitted,matplotlib,matplotlib.pyplot,sys
import numpy as np

# Define the processing steps
steps = [
    'ypbpr',
    'rowconv',
    'gradient',
    'matchedfilters',
    'matches',
    'features',
    'accuracyspectrum',
    'bestclassifier'
    
]
# plotter
class Plotter(object):
    '''Generate a series of plots that are useful for debugging your submitted code.'''
    def __init__(self,outputdirectory):
        '''Create the specified output directory, and initialize plotter to put its plots there'''
        os.makedirs(outputdirectory,exist_ok=True)
        self.figs = [ matplotlib.pyplot.figure(n) for n in range(0,10) ]
        self.outputdirectory = outputdirectory

    def make_plots(self, classes, nperclass):
        '''Create a new dataset object, run all the steps, and make all the plots'''
        self.output_filename=os.path.join(self.outputdirectory,'%s_vs_%s_%dtoks'%(classes[0],classes[1],nperclass))
        self.dataset = submitted.Dataset(classes, nperclass)
        for n in range(0,8):
            getattr(self.dataset, 'set_' + steps[n])()
            plotter_method = getattr(self, 'plot_%s'%(steps[n]))
            self.figs[n].clf()
            plotter_method(self.figs[n], self.dataset)                
            self.figs[n].savefig(self.output_filename + '_step%d.png'%(n))
        
    def plot_ypbpr(self, f, dataset):
        '''Plot the Y, Pr and Pb color planes of the first and last two images'''
        aa = f.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
        for row in range(3):
            aa[row,0].imshow(dataset.ypbpr[:,:,row,0],cmap='gray')
            aa[row,1].imshow(dataset.ypbpr[:,:,row,-1],cmap='gray')
        aa[0,0].set_title('YPbPr image 0')
        aa[0,1].set_title('YPbPr image %d'%(dataset.ndata-1))

    def plot_rowconv(self, f, dataset):
        '''Plot the diff and ave row-convolutions of the first image in the dataset'''
        aa = f.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
        for row in range(3):
            for col in range(2):
                aa[row,col].imshow(dataset.rowconv[:,:,3*col+row,0],cmap='gray')
        aa[0,0].set_title('Image 0 row diff')
        aa[0,1].set_title('Image 0 row ave')

    def plot_gradient(self, f, dataset):
        '''Plot Gx and Gy images of first image in the dataset'''
        aa = f.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
        for row in range(3):
            for col in range(2):
                aa[row,col].imshow(dataset.gradient[:,:,3*col+row,0],cmap='gray')
        aa[0,0].set_title('Image 0 Gx')
        aa[0,1].set_title('Image 0 Gy')

    def plot_matchedfilters(self, f, dataset):
        '''Plot the matchedfilters'''
        aa = f.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
        for col in range(2):
            for row in range(3):
                aa[row,col].imshow(dataset.matchedfilters[:,:,row,col],cmap='gray')
                aa[row,col].imshow(dataset.matchedfilters[:,:,row,col],cmap='gray')
        aa[0,0].set_title('Matched Filter 0')
        aa[0,1].set_title('Matched Filter 1')
        
    def plot_matches(self, f, dataset):
        '''Plot the match images of the first image in the dataset'''
        aa = f.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
        for col in range(2):
            for row in range(3):
                aa[row,col].imshow(dataset.matches[:,:,row,0,col],cmap='gray')
                aa[row,col].imshow(dataset.matches[:,:,row,0,col],cmap='gray')
        aa[0,0].set_title('Image 0 Match 0')
        aa[0,1].set_title('Image 0 Match 1')
        
    def plot_features(self, f, dataset):
        '''Plot scatter plot of the three features, with colors for the classes'''
        aa = f.subplots(nrows=2,ncols=1)
        sc="bgrcmyk"
        fnames=['Color','Gradient','Matched']
        for r in range(0,2):
            aa[r].scatter(x=dataset.features[:,r],y=dataset.features[:,2],c=[sc[p] for p in dataset.labels])
            aa[r].set_ylabel('%s Feat'%(fnames[r]))
        aa[1].set_xlabel('%s Feat'%(fnames[2]))

    def plot_accuracyspectrum(self, f, dataset):
        aa = f.subplots(nrows=3,ncols=1)
        fnames=['Color','Gradient','Matched']
        for r in range(3):
            aa[r].scatter(x=dataset.features[:,r],y=dataset.accuracyspectrum[:,r])
            aa[r].set_ylabel('Acc: %s Feat'%(fnames[r]))
        aa[2].set_xlabel('Feature Thresholds')
            
    def plot_bestclassifier(self, f, dataset):
        '''Plot scatter plot of the three features, with line for each best classifier'''
        aa = f.subplots(nrows=2,ncols=1)
        sc="bgrcmyk"
        fnames=['Color','Gradient','Matched']
        for r in range(0,2):
            aa[r].scatter(y=dataset.features[:,r],x=dataset.features[:,2],c=[sc[p] for p in dataset.labels])
            aa[r].plot([dataset.bestclassifier[2,0],dataset.bestclassifier[2,0]+1e-6],
                       [min(dataset.features[:,r]),max(dataset.features[:,r])])
            aa[r].plot([min(dataset.features[:,2]),max(dataset.features[:,2])],
                       [dataset.bestclassifier[r,0],dataset.bestclassifier[r,0]])
            aa[r].set_ylabel('%s Feat Acc=%d%%'%(fnames[r],round(100*dataset.bestclassifier[r,2])))
        aa[1].set_xlabel('%s Feat Acc=%d%%'%(fnames[2],round(100*dataset.bestclassifier[2,2])))

#####################################################
# If this is called from the command line,
# create a plotter with the specified arguments (or with default arguments),
# then create the corresponding plots in the 'make_cool_plots_outputs' directory.
if __name__ == '__main__':
    classes=['fire','water'] if len(sys.argv) < 3 else [sys.argv[1],sys.argv[2]]
    nperclass = 12 if len(sys.argv) < 4 else sys.argv[4]

    plotter=Plotter('make_cool_plots_outputs')
    plotter.make_plots(classes, nperclass)
    
