'''
The purpose of this class is to find the minimal energy configuration using Baysian Optomization.
Although this would normally require a continuous domain, we can use floor and ceiling functions to effectively discritize the domain into array indicies.
'''

from bayes_opt import BayesianOptimization
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Bayes_premade: # this class uses a premade Bayesian Optomization package - mostly just a proof of concept
    '''
    There are a few drawbacks to using the premade optomizer
        1) FIXED - it does not draw from a lattice, but instead from R^n as its domain - we need to clip the chosen values to integers in the objective function
        2) it does not respect the uniqueness of tiles and thus the same tile could be chosen multiple times in the optomal solution
    '''
    def __init__(self, grid, dict_list, cached_energies, init_points = 5, n_iter = 25):
        self.grid : np.ndarray = grid
        self.grid_shape = grid.shape
        self.num_tiles : int = self.grid_shape[0] * self.grid_shape[1] # number of tiles in the puzzle
        self.tile_data = dict_list
        self.cached_energies = cached_energies
        self.objective_function = self.create_objective_function()
        self.pbounds = { "position%i"%i : (0, self.num_tiles, int) for i in range(self.num_tiles)} # dictionary indicating the restrictions on parameter space over which to optomize

        self.optomizer = BayesianOptimization(f = self.objective_function, pbounds = self.pbounds, verbose=0, random_state = 1)

        self.optimal_params = self.optomizer.maximize(init_points,n_iter)

    def create_objective_function(self): # create the function that that the gausian processes will attempt to maximize

        def objective_function(**Array_Indicies): # the negative of the energy - the process maximizes but we want the minimal energy
            # Array_Indicies will be a dictionary with the same keys as self.pbounds
            

            return 

        return objective_function
    

class Bayes_custom: # this class contains an original Beysian optomizer specifically for the use herein where we only care about lattice points
    def __init__(self, grid, dict_list, cached_energies, init_points = 5, n_iter = 25):
        self.grid = grid
        self.grid_shape = grid.shape
        self.num_tiles = self.grid_shape[0] * self.grid_shape[1] # number of tiles in the puzzle
        self.tile_data = dict_list
        self.cached_energies = cached_energies
        self.objective_function = self.create_objective_function()

    def create_objective_function(self): # create the function that that the gausian processes will attempt to maximize
        # Since the Bayesian optomizer will use the real numbers as its domain, we will need to clip them to the lattice grid 

        def objective_function(): # the negative of the energy - the process maximizes but we want the minimal energy
            pass

        return objective_function