__authors__ = ['1706732', '1707531', '1704706']
__group__ = '14'

import numpy as np
import math
import operator
import utils
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        Initializes the train data
        Args:
            train_data: PxMxNx3 matrix corresponding to P color images
        Return: 
            assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if len(train_data.shape) == 4 and train_data.shape[-1] == 3:
            # Hem de convertir a grayscale
                train_data = utils.rgb2gray(train_data)
                
        self.train_data = np.array(train_data.reshape((train_data.shape[0], -1)), dtype="float")

    
    def _init_train_flip(self, train_data):
        """
        Initializes the train data
        Args:
            train_data: PxMxNx3 matrix corresponding to P color images
        Return: 
            assigns the train set (flipped horizontally) to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if len(train_data.shape) == 3:
            # Already in Grayscale
            if train_data.shape[-1] == 3: 
                # If RGB, convert to grayscale 
                train_data = utils.rgb2gray(train_data) 
            # Flip the images horizontally
            train_data_flipped = np.flip(train_data, axis=1)
            train_data_flipped = np.array(train_data_flipped.reshape((train_data_flipped.shape[0], -1)), dtype="float")
            train_data = np.array(train_data.reshape((train_data.shape[0], -1)), dtype="float")
            self.train_data = np.concatenate((train_data, train_data_flipped), axis=0)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        
        test_data = np.array(test_data, dtype = float)
        
        if len(test_data.shape) == 4 and test_data.shape[-1] == 3:
            # Hem de convertir a grayscale
                test_data = utils.rgb2gray(test_data)
        
        if len(test_data.shape) > 2:
            test_data = test_data.reshape(test_data.shape[0], -1)
            
        distancies = cdist(test_data, self.train_data, metric='euclidean')
        
        mespropers_ind = np.argpartition(distancies, k, axis=1)[:, :k]
        self.neighbors = self.labels[mespropers_ind]
        
        fila_ind = np.arange(distancies.shape[0])[:, None]
        distancies_veins = distancies[fila_ind, mespropers_ind]
        
        ind_ordenats = np.argsort(distancies_veins, axis=1)
        self.neighbors = np.take_along_axis(self.neighbors, ind_ordenats, axis=1)
        

    def get_class(self):
        """
        Get the class by maximum voting
        Args:
            None
        Return:
            1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #m = max elements
        #c = neighbour class
        
        neighbour_class = []    
        for neighbors in self.neighbors:
            #buscamos las clases sin repeticiones, los indices de cada clase y la cantidad de veces que se repite cada clase
            clas, index, inverse, counts = np.unique(neighbors, return_index=True, return_inverse=True, return_counts=True)
            #posiciones donde count es maximo
            m = np.where(counts == np.max(counts))[0]
            #cojemos el neighbor con el indice mas pequeño de entre los que count es maximo
            c = neighbors[index[m].min()]
            #añadimos a la lista
            neighbour_class.append(c)
        return np.array(neighbour_class)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
