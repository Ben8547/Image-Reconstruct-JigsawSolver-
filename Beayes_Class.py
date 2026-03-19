'''
The purpose of this class is to find the minimal energy configuration using Baysian Optomization.
Although this would normally require a continuous domain, we can use floor and ceiling functions to effectively discritize the domain into array indicies.
'''

from bayes_opt import BayesianOptimization
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Bayes:
    def __init__(self, grid, dict_list, cached_energies):
        self.grid = grid
        self.tile_data = dict_list
        self.cached_energies = cached_energies