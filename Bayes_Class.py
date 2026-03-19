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
        3) We cannot set the eneriges of the same tile to infinite
    '''
    def __init__(self, grid, dict_list, cached_energies, init_points = 5, n_iter = 25):
        self.grid : np.ndarray = grid
        self.grid_shape = grid.shape
        self.num_tiles : int = self.grid_shape[0] * self.grid_shape[1] # number of tiles in the puzzle
        self.tile_data = dict_list
        self.cached_energies = cached_energies
        self.objective_function = self.create_objective_function()
        self.pbounds = { "position%i"%i : (0, self.num_tiles-1, int) for i in range(self.num_tiles)} # dictionary indicating the restrictions on parameter space over which to optomize

        self.optomizer = BayesianOptimization(f = self.objective_function, pbounds = self.pbounds, verbose=0, random_state = 1)

        self.optomizer.maximize(init_points, n_iter)

        self.solution = np.zeros((self.grid_shape[0],self.grid_shape[1]), dtype=int)

        for i in range(self.num_tiles):
                m = i // self.grid_shape[1] # row index
                n = i % self.grid_shape[1] # column index
                self.solution[m,n] = self.optomizer.max["params"]["position%i"%i]

        self.energy = self.optomizer.max["target"]

    def create_objective_function(self): # create the function that that the gausian processes will attempt to maximize

        def objective_function(**Array_Indicies): # the negative of the energy - the process maximizes but we want the minimal energy
            # Array_Indicies will be a dictionary with the same keys as self.pbounds
            grid = np.zeros((self.grid_shape[0],self.grid_shape[1]), dtype=int)
            for i in range(self.num_tiles):
                m = i // self.grid_shape[1] # row index
                n = i % self.grid_shape[1] # column index
                grid[m,n] = Array_Indicies["position%i"%i]
            
            top_current = grid[1:, :]

            left_current = grid[:,1:]
        
            top_neighbors  = grid[:-1, :] # returns the matrix from the topmost row the second to penultimate row; these will all have their bottoms measured
            left_neighbors = grid[:, :-1]

            energy_top = np.sum(self.cached_energies[top_current, top_neighbors, 0])
            energy_left = np.sum(self.cached_energies[left_current, left_neighbors, 1])

            return energy_left + energy_top

        return objective_function
    

#
#
#------------------------------------------
#
# I/O  Functions
#
#------------------------------------------
#

def generate_bayes_from_file(filename="Inputs/Squirrel_Puzzle.jpg", grid_size=(8,8), color=True, energy_function = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y)), init_points = 5, n_iter = 25) -> Bayes_premade:
    """
    Create a simulation_grid object directly from an image file.

    filename : string : path of the file to be imported
    grid_size : tuple of integers : gives the number of puzzle pieces which the image is split into
    color : boolean : Determines if the image is read in color or black and white
    """

    num_tiles = grid_size[0]*grid_size[1] # total number of tiles in the image

    if color:
        color_volume = cv2.imread(filename, cv2.IMREAD_COLOR)# we need these to be int16 so that there is not wrapping of the unsinged integers when we compute energies. This should allow proper energy computation; commented out the int16 part because I ordered the terms in the mean
        #print(type(color_volume[0,0,0])) # it is indeed stored as uint8
        '''
        This outputs a 3 dimensional array: (hight, width, 3)
        We will need to store data differently taking this into account.
        '''
    else:
        gray_matrix = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    '''Chop up the image'''

    tiles = [] # this only contains the 

    if not color:
        width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
        length = gray_matrix.shape[0]
        tile_width = width//grid_size[1]
        tile_length = length//grid_size[0]
        
        tile_sides = np.empty( ( num_tiles, 4, max(tile_length,tile_width) ), dtype=np.float32)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tiles.append(gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)]) # need this one to reconstruct the array later
                index = i * grid_size[1] + j
                tile_sides[index, 0, :tile_width] = gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)]
                tile_sides[index, 2, :tile_width] = gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)]
                tile_sides[index, 1, :tile_length] = gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j]
                tile_sides[index, 3, :tile_length] = gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1]
        del gray_matrix # no need to store a large matrix any longer than we need it. We only need the boarders anyway
        cached_energies = cache_energies_grayscale(num_tiles, tile_sides, energy_function, tile_length,tile_width)

    else:
        width = color_volume.shape[1] # horizontal distance - should be the shoter of the two
        length = color_volume.shape[0]
        tile_width = width//grid_size[1]
        tile_length = length//grid_size[0]

        tile_sides = np.empty( ( num_tiles, 4, max(tile_length,tile_width), 3 ), dtype=np.float32)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tiles.append(color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:]) # need this one to reconstruct the array later
                index = i * grid_size[1] + j
                tile_sides[index, 0, :tile_width, :] = color_volume[tile_length*i,tile_width*j:tile_width*(j+1), :]
                tile_sides[index, 2, :tile_width, :] = color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1), :]
                tile_sides[index, 1, :tile_length, :] = color_volume[tile_length*i:tile_length*(i+1),tile_width*j, :]
                tile_sides[index, 3, :tile_length, :] = color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1, :]

        del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

        cached_energies = cache_energies_color(num_tiles, tile_sides, energy_function, tile_length,tile_width)

    grid = np.arange(0,num_tiles,1,dtype=int).reshape((grid_size[0],grid_size[1])) # the representation of the image; using uint8 because nothing is negative or bigger than 255 and thus using any other integer system would be wasteful


    tiles = np.array(tiles, dtype=object) # apparently you can make a list of arrays into an array - this makes indexing later much easier - this is a change from the previous version

    print("cached tile energies")
    

    return Bayes_premade(grid,tiles,cached_energies,init_points,n_iter)


def cache_energies_color(num_tiles, tile_sides, energyFunction, tile_length,tile_width):

    cached_energies = np.zeros((num_tiles,num_tiles,4),dtype=np.float32)

    for i in range(num_tiles): # run the first loop in parallel - innner loops apparently cannot be parallelized
        for j in range(num_tiles):
            cached_energies[i,j,0] = energyFunction( tile_sides[i, 0, :tile_width], tile_sides[j, 2, :tile_width,:] ) # cutting off at tile_width shouldn't matter since they subtract out anyway, but in the case of some other energy function this is a good practice to do anyway
            cached_energies[i,j,1] = energyFunction( tile_sides[i, 1, :tile_length], tile_sides[j, 3, :tile_length,:] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies


def cache_energies_grayscale(num_tiles, tile_sides, energyFunction, tile_length, tile_width):

    cached_energies = np.empty((num_tiles,num_tiles,4),dtype=np.float32)

    for i in range(num_tiles): # run the first loop in parallel - innner loops apparently cannot be parallelized
        for j in range(num_tiles):
            cached_energies[i,j,0] = energyFunction( tile_sides[i, 0, :tile_width], tile_sides[j, 2, :tile_width] ) # cutting off at tile_width shouldn't matter since they subtract out anyway, but in the case of some other energy function this is a good practice to do anyway
            cached_energies[i,j,1] = energyFunction( tile_sides[i, 1, :tile_length], tile_sides[j, 3, :tile_length] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies


def bayes_reconstruct(simulation : Bayes_premade, color = True):
    tile_width = (simulation.tile_data[0].shape[1]) # length of the top of an arbitrary tile
    tile_length = (simulation.tile_data[0].shape[0]) # length of the left of an arbitrary tile

    if color:
        width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
        length = tile_length * simulation.grid_shape[0]

        resotred_page = np.zeros((length,width,3))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.solution[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = simulation.tile_data[dict_index]
    else:

        width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
        length = tile_length * simulation.grid_shape[0]

        resotred_page = np.zeros((length,width))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.solution[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = simulation.tile_data[dict_index]

    return resotred_page.astype(np.uint8) # jpg can only handle this resolution anyway

def save_bayes_output(filename, simulation : Bayes_premade, color = True, reconstruction = None):
    if reconstruction is None:
        reconstruction = bayes_reconstruct(simulation,color)

    cv2.imwrite(filename, reconstruction)
