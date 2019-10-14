import numpy as np
import cmath,math,os,collections
from PIL import Image

people = ['Arnold_Schwarzenegger','George_HW_Bush','George_W_Bush','Jiang_Zemin']

class KNN(object):
    """KNN: a class that computes K-nearest-neighbors of each image in a dataset"""
    def __init__(self,datadir,nfeats,transformtype,K):
        self.nfeats = nfeats  # Number of features to keep
        self.transformtype = transformtype # Type of transform, 'dct' or 'pca'
        self.K = K # number of neighbors to use in deciding the class label of each token
        self.npeople = 4
        self.nimages = 12
        self.ndata = self.npeople*self.nimages
        self.nrows = 64
        self.ncols = 64
        self.npixels = self.nrows * self.ncols
        self.data = np.zeros((self.ndata,self.nrows,self.ncols),dtype='float64')
        self.labels = np.zeros(self.ndata, dtype='int')
        for person in range(0,self.npeople):
            for num in range(0,self.nimages):
                datum = 12*person + num
                datafile = os.path.join(datadir,'%s_%4.4d.ppm'%(people[person],num+1))
                img = np.asarray(Image.open(datafile))
                bw_img = np.average(img,axis=2)
                self.data[datum,:,:] = bw_img
                self.labels[datum] = person
        
    # PROBLEM 2.0
    # set_vectors - reshape self.data into self.vectors.
    #   Vector should scan the image in row-major ('C') order, i.e.,
    #   self.vectors[datum,n1*ncols+n2] = self.data[datum,n1,n2]
    def set_vectors(self):  #flat
        self.vectors = np.zeros((self.ndata,self.nrows*self.ncols),dtype='float64')
        #for d in range(0,self.ndata):
        #    self.vectors[d,:] = self.data.flatten[d,:,:]
        for d in range(0,self.ndata):
            for n1 in range(0,self.nrows):
                for n2 in range(0,self.ncols):
                    self.vectors[d,n1*self.ncols+n2] = self.data[d,n1,n2]
        
        # TODO: fill self.vectors


    # PROBLEM 2.1
    # set_mean - find the global mean image vector
    # mean of each pixels and put into a vector
    def set_mean(self):
        self.mean = np.zeros(self.npixels, dtype='float64')
        #self.vectors 長成 [ndata, ncol*nrow] = [ndata, npixels]
        #self.npixels = self.nrows * self.ncols
        for n in range(0, self.npixels):
            self.mean[n] = np.sum(self.vectors[:,n])/self.ndata
        #print(self.mean)
        # TODO: fill self.mean

    # PROBLEM 2.2
    # set_centered - compute the zero-centered dataset, i.e., subtract the mean from each vector.
    def set_centered(self):
        self.centered = np.zeros((self.ndata,self.npixels), dtype='float64')
        for n in range(0, self.npixels):
            self.centered[:,n] = self.vectors[:,n] - self.mean[n]
        # TODO: fill self.centered
            
    # PROBLEM 2.3
    #
    # set_transform - compute the feature transform matrix (DCT or PCA)
    #  If transformtype=='dct':
    #    transform[ktot,ntot] = basis[k1,n1,nrows] * basis[k2,n2,ncols]
    #    basis[k1,n1,nrows] = (D/sqrt(nrows)) * cos(pi*(n1+0.5)*k1/nrows)
    #    D = 1 if k1==0, otherwise D = sqrt(2)
    #    Image pixels are scanned in row-major order ('C' order), thus ntot = n1*ncols + n2.
    #    Frequencies are scanned in diagonal L2R order: (k1,k2)=(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),...
    #  If transformtype=='pca':
    #    self.transform[k,:] is the unit-norm, positive-first-element basis-vector in the
    #       principal component direction with the k'th highest eigenvalue.
    #    You can get these from eigen-analysis of covariance or gram, or by SVD of the data matrix.
    #    To pass the autograder, you must check the sign of self.transform[k,0], for each k,
    #    and set self.transform[k,:] = -self.transform[k,:] if self.transform[k,0] < 0.
    def set_transform(self):
        if self.transformtype=='dct':
            self.transform = np.zeros((self.nfeats,self.npixels),dtype='float64')
            # TODO: set self.transform in the DCT case
            # create k_all 
            dim = int(math.sqrt(self.nfeats))   # how basis looks like
            k_all = []
            for ktot in range(dim):
                for k2 in range(dim+1):
                    for k1 in range(dim+1):
                        if k1+k2==ktot:
                            k_all.append([k1,k2])

            for ktot in range(dim,0,-1):
                for k2 in range(dim,0,-1):
                    for k1 in range(dim,0,-1):
                        if k1+k2==ktot:
                            k_all.append([dim-k1,dim-k2])
            for k in range(self.nfeats):
                k1, k2 = k_all[k]
                for n1 in range(self.nrows):
                    for n2 in range(self.ncols):
                        ntot = n1* self.ncols + n2
                        if k1 == 0:
                            D = 1
                        else:
                            D = np.sqrt(2)
                        basis1 = (D/np.sqrt(self.nrows))*np.cos(np.pi*(n1+0.5)*k1/self.nrows)
                        if k2 == 0:
                            D = 1
                        else:
                            D = np.sqrt(2)
                        basis2 = (D/np.sqrt(self.ncols))*np.cos(np.pi*(n2+0.5)*k2/self.nrows)
                        self.transform[k,ntot] = basis1*basis2            
        elif self.transformtype=='pca':
            self.transform = np.zeros((self.nfeats,self.npixels),dtype='float64')
            #k = nfeat non-full matrices
            #ndata<npixels X = M*D 
            U, S, Vh = np.linalg.svd(self.centered, full_matrices = False)
            # Vh are put in rows, and is sorted with descending 
            for k in range(self.nfeats):
                self.transform[ k , : ] = Vh[ k , : ]
                if self.transform[ k , 0 ] < 0:
                    self.transform[ k , : ] = -self.transform[ k , : ]
            # TODO: set self.transform in the PCA case


    # PROBLEM 2.4
    # set_features - transform the centered dataset to generate feature vectors.
    # 這題在求y = Vh*X
    # Vh似乎就是上一題求出的self.transform
    def set_features(self):
        self.features = np.zeros((self.ndata,self.nfeats),dtype='float64')
        Vh = self.centered   #self.ndata*self.npixels
        Xh = self.transform  #self.nfeats*self.npixels
        X = np.transpose(Xh)    
        self.features = np.dot(Vh, X)  #self.ndata*self.nfeats
        # TODO: fill self.features

    # PROBLEM 2.5
    # set_energyspectrum: the fraction of total centered-dataset variance explained, cumulatively,
    #   by all feature dimensions up to dimension k, for 0<=k<nfeats.
    def set_energyspectrum(self):
        self.energyspectrum = np.zeros(self.nfeats,dtype='float64')
        # TODO: calculate total dataset variances, then set self.energyspectrum
        denom = np.sum(np.multiply(self.centered, self.centered))
        yy = np.multiply(self.features, self.features)
        eigenvalues = np.sum(yy, axis = 0)  #row and row adding
        
        self.energyspectrum[0] = eigenvalues[0]
        for k in range(1, self.nfeats):
            self.energyspectrum[k] = self.energyspectrum[k-1] + eigenvalues[k]
        self.energyspectrum = self.energyspectrum/denom

    # PROBLEM 2.6
    #
    # set_neighbors - indices of the K nearest neighbors of each feature vector (not including itself).
    #    return: a matrix of datum indices, i.e.,
    #    self.features[self.neighbors[n,k],:] should be the k'th closest vector to self.features[n,:].
    def set_neighbors(self):
        self.neighbors = np.zeros((self.ndata,self.K), dtype='int')
        # TODO: fill self.neighbors
        dist = np.zeros((self.ndata, self.ndata), dtype = 'float64')
        for i in range(self.ndata):
            for j in range(self.ndata):
                distance = self.features[i,:] - self.features[j,:]
                dist[i,j] = np.linalg.norm(distance)
        dist_ordered = np.sort(dist)    #把距離從小到大排序
        nn = np.zeros((self.ndata, self.ndata), dtype = 'int') 
        # coord 儲存從order_dist map 回到dist 的座標
        #coord = np.zeros((self.ndata, self.K), dtype = 'int') 
        for i in range(self.ndata):
            for j in range(1, self.K + 1):
                index = np.argwhere(dist[i,:] == dist_ordered[i,j])
                nn[i,j] = int(index[0][0])
                #coord[i,j-1] = np.argwhere( dist == dist_ordered[i,j])
                #coord[i,j-1][0][0] = self.neighbors[i,j-1]
        self.neighbors = nn[:,1:(self.K+1)]
                
        
    # PROBLEM 2.7
    #
    # set_hypotheses - K-nearest-neighbors vote, to choose a hypothesis person label for each datum.
    #   If K>1, then check for ties!  If the vote is a tie, then back off to 1NN among the tied options.
    #   In other words, among the tied options, choose the one that has an image vector closest to
    #   the datum being tested.
    def set_hypotheses(self):
        self.hypotheses = np.zeros(self.ndata, dtype='int')
        #self.labels = np.zeros(self.ndata, dtype='int')
        #cnt = 0
        n_people = self.neighbors   #存取轉成人物編號的neighbor
        # 從第一張圖片開始
        for n in range(self.ndata):
            #創一個array 存取不同人物的票數統計
            sum_vote = np.zeros(self.npeople, dtype = 'int')
            #先把neighbors 裡面 圖片的編號轉換成人物編號
            for i in range(self.K):
                num_people = self.labels[self.neighbors[n,i]]   #圖片編號轉成人物編號
                n_people[n,i] = num_people    #人物編號登記到矩陣裡
                sum_vote[num_people] += 1    #然後統計各個人的票數
            h_vote = max(sum_vote)  #找出最高票數
            
            tie = 0     #看有沒有重複票數的參數
            for p in range(0,len(sum_vote)):
                if sum_vote[p] == h_vote:
                    tie += 1
            if tie == 1:     #no tie
                self.hypotheses[n] = np.argmax(sum_vote)
            else:   #tie -> 1NN
                self.hypotheses[n] = n_people[n,0]  #1NN
        # TODO: fill self.hypotheses

    # PROBLEM 2.8
    #
    # set_confusion - compute the confusion matrix
    #   confusion[r,h] = number of images of person r that were classified as person h    
    def set_confusion(self):
        self.confusion = np.zeros((self.npeople,self.npeople), dtype='int')
        # TODO: fill self.confusion
        for n in range(self.ndata):
            ref = self.labels[n]    #原始圖片的人物編號
            hyp = self.hypotheses[n]    #猜測的結果
            self.confusion[ref, hyp] += 1
        
    # PROBLEM 2.9
    #

    # set_metrics - set self.metrics = [ accuracy, recall, precision ]
    #   recall is the average, across all people, of the recall rate for that person
    #   precision is the average, across all people, of the precision rate for that person
    def set_metrics(self):
        self.metrics = [1.0, 1.0, 1.0]  # probably not the correct values!
        # TODO: fill self.metrics
        
        #Accuracy
        A_d = np.sum(self.confusion)
        A_n = np.sum(self.confusion[i][i] for i in range(self.npeople))
        print(A_d)
        print(A_n)
        A = A_n/A_d
        
        #Recall
        R_d = np.sum(self.confusion, axis = 1)
        R_n = []
        for r in range(self.npeople):
            R_n.append(self.confusion[r,r])
        R = np.sum(np.divide(R_n, R_d))/4  
    
        #Precision
        P_d = np.sum(self.confusion, axis = 0)
        P_n = []
        for h in range(self.npeople):
            P_n.append(self.confusion[h,h])
        P = np.sum(np.divide(P_n, P_d))/4

        self.metrics = [A,R,P]
    # do_all_steps:
    #   This function is given, here, in case you want to do all of the steps all at once.
    #   To do that, you can type
    #   knn=KNN('data',36,'dct',4)
    #   knn.do_all_steps()
    #
    def do_all_steps(self):
        self.set_vectors()
        self.set_mean()
        self.set_centered()
        self.set_transform()
        self.set_features()
        self.set_energyspectrum()
        self.set_neighbors()
        self.set_hypotheses()
        self.set_confusion()
        self.set_metrics()

