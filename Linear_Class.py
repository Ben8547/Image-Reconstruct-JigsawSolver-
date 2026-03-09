'''
This class serves as a recreation of Rui Yu et al.'s linear programming puzzle solver.
'''
import numpy as np
import cv2
from numba import njit, prange
import warp as wp

#
#------------------------------------------
#
# Helper functions (using warp for parallelization)
#
#------------------------------------------
#

@wp.kernel
def compute_weight_row(min_energies_top: wp.array, min_energies_left: wp.array, num_tiles: wp.array, weights: wp.array):
    row = wp.tid() #thread index
    weights[row,:,0] = wp.min(min_energies_top[row + num_tiles], min_energies_top)
    weights[row,:,1] = wp.min(min_energies_left[row + num_tiles], min_energies_left)

    

#
#------------------------------------------
#
# Main Solver
#
#------------------------------------------
#


class Linear_solver:
    def __init__(self, grid: np.ndarray, cached_energy: np.ndarray ):
        self.grid = grid
        self.cached_energy = wp.array(cached_energy,dtype=wp.float32)
        self.weights = self.compute_weights()

    def  compute_weights(self):
        '''w_{ijo} will require an array size of the energies to store'''
        num_tiles = self.cached_energy.shape[0]
        weights = wp.empty((num_tiles,num_tiles,2) , dtype=wp.float32)
        # find the minimal interactions along each direction
        min_energies_top_1 = np.min(self.cached_energy[:,:,0],axis = 0) # the jth element of the resultant array gives the value min_{k\neq j}(E_{kj0})
        min_energies_left_1 = np.min(self.cached_energy[:,:,1],axis = 0) # the jth element of the resultant array gives the value min_{k\neq j}(E_{kj1})
        min_energies_top_2 = np.min(self.cached_energy[:,:,0],axis = 1) # the ith element of the resultant array gives the value min_{k\neq j}(E_{ik0})
        min_energies_left_2 = np.min(self.cached_energy[:,:,1],axis = 1) # the ith element of the resultant array gives the value min_{k\neq j}(E_{ik1})
        # For reference:
        # min_energies_bottom_1 = min_energies_top_2
        # min_energies_right_1 = min_energies_left_2
        # min_energies_bottom_2 = min_energies_top_1
        # min_energies_right_2 = min_energies_left_1

        min_energies_top = wp.array(np.concatenate(min_energies_top_1,min_energies_top_2), dtype=wp.float32)
        min_energies_left = wp.array(np.concatenate(min_energies_left_1, min_energies_left_2), dtype=wp.float32)

        wp.launch(kernel = compute_weight_row, dim = num_tiles,inputs=[min_energies_top,min_energies_left,num_tiles,weights],outputs=[weights])
        return weights
    
    @njit(fastmath=True, parallel = True)
    def cost(self):
        for 
            x_cost = np.linalg.norm(,ord=0) #L0 norm
            y_cost = np.linalg.norm(,ord=0)
        return x_cost + y_cost




#
#------------------------------------------
#
# I/O  Functions
#
#------------------------------------------
#

def generate_linear_from_file(filename="Inputs/Squirrel_Puzzle.jpg", grid_size=(8,8), color=True, energy_function = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y)), T0=10., Tf=0.5, geometric_decay_rate=0.9999, updates=False, numberGenerations:int = 100, parentsPerGeneration:int = 4, populationSize:int = 1000) -> Linear_solver:
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
    
    return Linear_solver()


@njit(parallel = True, fastmath = True)
def cache_energies_color(num_tiles, tile_sides, energyFunction, tile_length,tile_width):

    cached_energies = np.zeros((num_tiles,num_tiles,4),dtype=np.float32)

    for i in prange(num_tiles): # run the first loop in parallel - innner loops apparently cannot be parallelized
        for j in range(num_tiles):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cached_energies[i,j,0] = np.inf
                    cached_energies[i,j,1] = np.inf
                else:
                    cached_energies[i,j,0] = energyFunction( tile_sides[i, 0, :tile_width], tile_sides[j, 2, :tile_width,:] ) # cutting off at tile_width shouldn't matter since they subtract out anyway, but in the case of some other energy function this is a good practice to do anyway
                    cached_energies[i,j,1] = energyFunction( tile_sides[i, 1, :tile_length], tile_sides[j, 3, :tile_length,:] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies


@njit(parallel = True, fastmath = True)
def cache_energies_grayscale(num_tiles, tile_sides, energyFunction, tile_length, tile_width):

    cached_energies = np.empty((num_tiles,num_tiles,4),dtype=np.float32)

    for i in prange(num_tiles): # run the first loop in parallel - innner loops apparently cannot be parallelized
        for j in range(num_tiles):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cached_energies[i,j,0] = np.inf
                    cached_energies[i,j,1] = np.inf
                else:
                    cached_energies[i,j,0] = energyFunction( tile_sides[i, 0, :tile_width], tile_sides[j, 2, :tile_width] ) # cutting off at tile_width shouldn't matter since they subtract out anyway, but in the case of some other energy function this is a good practice to do anyway
                    cached_energies[i,j,1] = energyFunction( tile_sides[i, 1, :tile_length], tile_sides[j, 3, :tile_length] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies


def linear_reconstruct(simulation : Linear_solver, color = True):
    tile_width = (simulation.tile_data[0].shape[1]) # length of the top of an arbitrary tile
    tile_length = (simulation.tile_data[0].shape[0]) # length of the left of an arbitrary tile

    if color:
        width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
        length = tile_length * simulation.grid_shape[0]

        resotred_page = np.zeros((length,width,3))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.product[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = simulation.tile_data[dict_index]
    else:

        width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
        length = tile_length * simulation.grid_shape[0]

        resotred_page = np.zeros((length,width))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.product[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = simulation.tile_data[dict_index]

    return resotred_page.astype(np.uint8) # jpg can only handle this resolution anyway

def save_linear_output(filename, simulation : Linear_solver, color = True, reconstruction = None):
    if reconstruction is None:
        reconstruction = linear_reconstruct(simulation,color)

    cv2.imwrite(filename, reconstruction)