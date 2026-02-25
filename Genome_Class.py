import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep
from Annealing_Class import simulation_grid

#
#
#------------------------------------------
#
# Genome Class
#
#------------------------------------------
#

class Genome:
    '''
    This class runs the details of the genetic algorithm
    '''
    def __init__(self, grid, dict_list, cached_energies, numberGenerations:int = 100, parentsPerGeneration:int = 4, populationSize:int = 1000, T0:float=10., Tf:float=0.5, geometric_decay_rate:float=0.9999):
        self.initial_grid = grid # the seed grid - usually just the input puzzle image
        self.tile_data = dict_list # see the simulation_grid class
        self.cached_energies = cached_energies # see the simulation_grid class

        self.numberGenerations = numberGenerations # number of generations to run in the simulation
        self.paraentsPerGen = parentsPerGeneration  # number of indeviduals to keep from the old generation at the begining of each new generation
        self.populationSize = populationSize # number of individuals in each generation
        self.chromosomes = np.empty(self.populationSize,dtype=object) # array of chromosome arrays; a chromosome is a permutation of the initial grid (a member of solution space  )


        self.T0 = T0 # initial annealing parameter
        self.Tf = Tf # initial annealing parameter
        self.cooling_rate = geometric_decay_rate # initial annealing parameter
    
    def total_energy_grid(self,grid) -> float: 
        top_current = grid[1:, :]
        left_current = grid[:,1:]
    
        top_neighbors  = grid[:-1, :] # returns the matrix from the topmost row the second to penultimate row; these will all have their bottoms measured
        left_neighbors = grid[:, :-1]

        energy_top = np.sum(self.cached_energies[top_current, top_neighbors, 0])
        energy_left = np.sum(self.cached_energies[left_current, left_neighbors, 1])

        return energy_top + energy_left
    
    def n_most_fit(self,n:int):
        if n > self.populationSize:
            raise(ValueError)
        
        list_of_fitnesses = []
        for i in range(self.populationSize):
            if self.chromosomes[i] == None:
                list_of_fitnesses.append(np.inf) # highest energy, worst fitness
            else:
                list_of_fitnesses.append(self.total_energy_grid(self.chromosomes[i]))
        
        self.chromosomes = self.chromosomes[np.argsort(list_of_fitnesses)] # order the chromosomes from most fit to least fit (least to highest energy)

        return self.chromosomes[:n] # return the first n items of the list; will fail if n > length of chromosomes hence the earlier error message
    
    class Child:
        '''This class takes members of the genome and recombines them to create children which are then added to the gene pool.
        The class in particular handles the generation of said children.'''
        def __init__(self, parent1, parent2, outerself):
            return

#
#
#------------------------------------------
#
# I/O  Functions
#
#------------------------------------------
#