__authors__ = ['1706732', '1707531', '0000000']
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
            indexs_aleatoris = np.random.choice(self.X.shape[0], size=self.K, replace=False)
            self.centroids = self.X[indexs_aleatoris]
            
        elif self.options['km_init'].lower() == 'custom':
            indexs_aleatoris = np.random.choice(self.X.shape[0], size=self.K, replace=False)
            self.centroids = self.X[indexs_aleatoris]
                        
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
        while 1:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges(): break;
            elif self.num_iter >= self.max_iter: break;
        
        
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        distancies = distance(self.X, self.centroids)
        
        dist_punt_centroid = pow(distancies[np.arange(len(self.X)), self.labels], 2)
        self.WCD = np.mean(dist_punt_centroid)
        return np.mean(dist_punt_centroid)
    
    
    

    def fisher(self):
        self.fisher_score = self.WCD / self.inter_clase
        return self.fisher_score

    def bestK_fisher(self, max_K):
        self.K = 2
        self.fit()

        self.withinClassDistance()
        self.inter_class()
        prev_fisher = self.fisher()

        k_index = 2
        max_diff = 0
        opt_K = 2
        while k_index < max_K:
            k_index += 1
            self.K = k_index
            self.fit()

            self.withinClassDistance()
            self.inter_class()
            current_fisher = self.fisher()

            diff = prev_fisher - current_fisher

            if diff > max_diff:
                max_diff = diff
                opt_K = self.K

            prev_fisher = current_fisher

        self.K = opt_K
        self.fit()
        return opt_K

    def find_bestK(self, max_K):
        if self.options['fitting'] == 'WCD':
            best_k = self.bestK_WCD(10)
        elif self.options['fitting'] == 'Fisher':
            best_k = self.bestK_fisher(10)
        return best_k

    def bestK_WCD(self, max_K):
        self.K = 2
        self.fit()
        prev_WCD = self.withinClassDistance()

        idx = 2
        reduction = 0
        while (idx < max_K) and ((100 - reduction) >= 20):
            idx += 1
            self.K = idx
            self.fit()
            curr_WCD = self.withinClassDistance()
            reduction = 100 * (curr_WCD / prev_WCD)
            prev_WCD = curr_WCD
        if idx != max_K:
            self.K = idx - 1
            self.fit()
            return idx
        else:
            return max_K


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
