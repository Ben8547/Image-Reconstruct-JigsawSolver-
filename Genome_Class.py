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
    def __init__(self, grid_list, dict_list, cached_energies, numberGenerations:int = 100, parentsPerGeneration:int = 4, populationSize:int = 1000, T0:float=10., Tf:float=0.5, geometric_decay_rate:float=0.9999, updates=False):
        if len(grid_list > populationSize):
            raise ValueError("grid_list cannot have more elements than populationSize")
        self.grid_shape = grid_list[0].shape # the seed grid - usually just the input puzzle image
        self.tile_data = dict_list # see the simulation_grid class
        self.cached_energies = cached_energies # see the simulation_grid class

        self.numberGenerations = numberGenerations # number of generations to run in the simulation
        self.paraentsPerGen = parentsPerGeneration  # number of indeviduals to keep from the old generation at the begining of each new generation
        self.populationSize = populationSize # number of individuals in each generation
        self.chromosomes = np.empty(self.populationSize,dtype=object) # array of chromosome arrays; a chromosome is a permutation of the initial grid (a member of solution space  )
        self.chromosomes[0:len(grid_list)] = grid_list # populate the chromosomes with any initial chromosomes

        self.mutation_probability = 1./1000. # probability of random mutation when creating a child
        self.T0 = T0 # initial annealing parameter
        self.Tf = Tf # initial annealing parameter
        self.cooling_rate = geometric_decay_rate # initial annealing parameter

        self.updates = updates # boolean that determines weather to print updates after each generation completes

    def run_simulation(self):
        self.initial_anneal()
        for generation in range(self.numberGenerations):
            self.chromosomes = self.n_most_fit(self.paraentsPerGen)
            if self.updates:
                print(f"Best starting energy of generation {generation} is {self.total_energy_grid(-1)}")
                print(f"Worst starting energy of generation {generation} is {self.total_energy_grid(0)}")
            for i in range(self.populationSize - self.paraentsPerGen):
                parents  = np.random.choice(self.chromosomes[:self.paraentsPerGen],replace=False)
                self.chromosomes[self.paraentsPerGen + i] = self.generate_child(parent1=parents[0],parent2=parents[1])

            self.chromosomes[0] = self.n_most_fit(1)
            self.chromosomes[1:] = np.empty(shape=self.populationSize-1,dtype=object)
            self.initial_anneal # the annealing won't disrupt a fully formed image, but if we are missing a few tile, the annealing might be able to fix it
        return self.chromosomes[0]

    def initial_anneal(self):
        ''' The genome class will perform well if matches have already been found, but our current algorithm is not great at finding said matches. Thus,
         We run annealing on the seed grids before the genetic algorithm in order to find these initial pairings; ideally there are repeats between parents '''
        for i in range(self.populationSize):
            if self.chromosomes[i] != None:
                annealed_chromosome = simulation_grid(self.chromosomes[i],self.tile_data,self.cached_energies,self.T0, self.Tf,self.cooling_rate)
                annealed_chromosome.anneal()
                self.chromosomes[i] = annealed_chromosome.simGrid
    
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
    
    def generate_child(self,parent1, parent2):
        grid = -np.ones((1,1),dtype=int) # begin with an empty 1x1 array
        '''seed the child with the first tile.'''
        parent_adjacencies_lookup, parent_adjacencies = self.search_parents_for_same_adjacencies(parent1, parent2)
        return
    
    def search_parents_for_same_adjacencies(self, parent1, parent2):
            '''
            In this function we search the parents for any equivalent adjacencies
            We return a list of dictionaries. Values of the dicts contain the adjacent member, if it is shared.
            We also need to store a list of all of the adjacencies so that we can choose one at random when filling the child. Repeats are fine - just think of it as weighting by the number of adjacencies that item has
            '''
            parent1 = self.parents[0].ravel() # it will probably be easier to parse the flat arrays; probably fine to use ravel since I don't edit anything; .ravel is faster than .flat
            parent2 = self.parents[1].ravel()
            cols = self.grid_shape[1]
            rows = self.grid_shape[0]

            list_of_tuples = [{"top": None, "bottom": None, "left": None, "right": None} for _ in range(len(parent1))] # if it's in the "top" slot then it means that it is to the left of the entry the dictionary represents
            list_of_adjacencies = []

            for i in range(len(parent1)//2): # we can skip every other entry and still read each adjacency. (or we could read each one and only consider left and top adjacencies but that is probably less efficient since most of the time probably comes from the element in parent two)
                value = parent1[i]
                parent2_arg = np.argwhere(  parent2 == value)[0,0] # should find the point where parent2 takes the same value as parent 1 - probably done as efficintly as possible since there is a numpy function doing the work.
                if (i % cols != 0 and parent2_arg % cols != 0): # conditions for not having a neighbor on a particular side; both are not on top row
                    if parent1[i-1]==parent2[parent2_arg-1]:
                        neighbor = parent1[i-1]
                        list_of_tuples[parent1[i]]["left"] = neighbor
                        list_of_tuples[neighbor]["right"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i >= cols and parent2_arg >= cols):
                    if parent1[i - cols] == parent2[parent2_arg - cols]:
                        neighbor = parent1[i-cols]
                        list_of_tuples[parent1[i]]["top"] = neighbor
                        list_of_tuples[neighbor]["bottom"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i % cols != cols-1) and (parent2_arg % cols != cols-1):
                    if parent1[i+1] == parent2[parent2_arg+1]:
                        neighbor = parent1[i+1]
                        list_of_tuples[parent1[i]]["right"] = neighbor
                        list_of_tuples[neighbor]["left"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i // cols != rows-1) and (parent2_arg // cols != rows-1):
                    if parent1[i+cols] == parent2[parent2_arg + cols]:
                        neighbor = parent1[i+cols]
                        list_of_tuples[parent1[i]]["bottom"] = neighbor
                        list_of_tuples[neighbor]["top"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

            return list_of_tuples, list_of_adjacencies

#
#
#------------------------------------------
#
# I/O  Functions
#
#------------------------------------------
#