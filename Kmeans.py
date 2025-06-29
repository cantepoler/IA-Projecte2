__authors__ = ['1706732', '1707531', '1704706']
__group__ = '14'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()
        self.max_iter = 500000
        self.WCD = None
        self.ICD = None
        

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        
        X = X.astype(float)
        if len(X.shape) > 2:
            F, C, D = X.shape
            X = X.reshape(F*C, D)
        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################    
        
        elements_unics = []
        if self.options['km_init'].lower() == 'first':
            for ind, fila in enumerate(self.X):
                if not any(np.array_equal(fila, element) for element in elements_unics):#Si la fila encara no ha estat triada:
                    elements_unics.append(fila)
                    if len(elements_unics) == self.K:
                        break;
            self.centroids = np.array(elements_unics)
            
        elif self.options['km_init'].lower() == 'random':
            X_unics = np.unique(self.X, axis=0)
            indexs_aleatoris = np.random.choice(X_unics.shape[0], size=self.K, replace=False)
            self.centroids = X_unics[indexs_aleatoris]
            
        elif self.options['km_init'].lower() == 'maxdist':
            centroides = [self.X[np.random.randint(self.X.shape[0])]] #El primer centroide és aleatori
            
            for i in range (1, self.K):                                 #Els següents són els que més lluny es troben dels centroides ja triats
                dists = distance(self.X, np.array(centroides))
                min_dists = np.min(dists, axis=1)
                seguent_centroide = np.argmax(min_dists)
                centroides.append(self.X[seguent_centroide])
            
            self.centroids = np.array(centroides)
            
                        
    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distancies = distance(self.X, self.centroids)
        
        self.labels = np.argmin(distancies, axis=1)
        
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid

        Args:
            None
        """
        self.old_centroids = np.copy(self.centroids)

        # Check if there is any cluster without points and calculate the new centroids
        if np.any(self.labels) > 0:
            for i in np.arange(self.K):
                self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        if (np.equal(self.centroids, self.old_centroids).all()):
            return True
        return False

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self.num_iter = 0
        self._init_centroids()
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            if self.converges():
                break
            self.num_iter +=1
        
        
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        distancies = distance(self.X, self.centroids)
        
        dist_punt_centroid = pow(distancies[np.arange(len(self.X)), self.labels], 2)
        self.WCD = np.mean(dist_punt_centroid)
        return np.mean(dist_punt_centroid)
    
    def interClassDistance(self):
        """
        Returns the inter class distance using centroid distances
        """
        suma = 0
        n = 0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                suma += dist
                n += 1
        self.ICD = suma/n
        return self.ICD       
        

    def fisher(self):
        self.withinClassDistance()
        self.interClassDistance()
        self.fisher_val = self.ICD/self.WCD
        return self.fisher_val


    def find_bestK(self, max_K):
        if self.options['fitting'] == 'WCD':
            best_k = self.bestK_min(max_K, 'WCD')
        elif self.options['fitting'] == 'ICD':
            best_k = self.bestK_max(max_K, 'ICD')
        elif self.options['fitting'] == 'Fisher':
            best_k = self.bestK_max(max_K, 'Fisher')
        return best_k

    def bestK_min(self, max_K, fitting):         #El bestK original
        minim = 20 #% minim acceptable
        best_K = max_K
        anterior = None
        
        wcds = []
        
        for k in range(2, max_K+1):
            kmeans = KMeans(self.X, K=k, options=self.options)
            kmeans.fit()
            WCD_k = kmeans.withinClassDistance()
                
            wcds.append(WCD_k)

            if anterior is not None:
                decrement = 100*(1-(WCD_k/anterior))    #Formula pdf
                if decrement < minim:
                    best_K = k-1
                    break
            
            anterior = WCD_k
            
        self.K = best_K
        return best_K
    

    def bestK_max(self, max_K, fitting):
        minim = 20  #% minim acceptable
        best_K = 2
        anterior = None
    
        for k in range(2, max_K+1):
            kmeans = KMeans(self.X, K=k, options=self.options)
            kmeans.fit()
            
            if (fitting == 'ICD'):
                heur_k = kmeans.interClassDistance()
                
            else:
                heur_k = kmeans.fisher()
    
            if anterior is not None:
                increment = 100 * ((heur_k-anterior)/anterior)
                if increment < minim:
                    best_K = k-1
                    break
    
            anterior = heur_k
            best_K = k
    
        self.K = best_K
        self.fit()
        return best_K



def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points 
        C (numpy array): KxD 2nd set of data points 
    """
    return np.linalg.norm(np.expand_dims(X, axis=1) - C, axis=2)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    
    probs = utils.get_color_prob(centroids)
    
    colors = np.argmax(probs, axis=1)
    
    return [utils.colors[i] for i in colors]
