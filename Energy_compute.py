import numpy as np
import cv2
#import warp as wp
from numba import njit, prange, jit
from typing import Callable

@njit(parallel = True, fastmath=True)
def compatability(x,y):
    return np.mean(np.abs(x-y))

def compute_energy(file,color=True, energyFunction: Callable = compatability, puzzle_shape = (8,8)):

    rows = puzzle_shape[0]
    columns = puzzle_shape[1]
    num_tiles = rows*columns

    grid = np.arange(0,num_tiles,1).reshape((rows,columns))

    '''Load in the test file'''
    if color:
        array = cv2.imread(file, cv2.IMREAD_COLOR)

        '''
        This outputs a 3 dimensional array: (hight, width, 3)
        We will need to store data differently taking this into account.
        '''
    else:
        array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    cached_energies = cache_energies(color,array,columns,rows, num_tiles, energyFunction)

    #Since we only did top and left, we can recover bottom and right since the matrix has the following property cache[i,j,0] = cache[j,i,2] and cache[i,j,1] = cache[j,i,3]
    # by only computing half of the directions in the loop we should halve the compute time of the loop

    X, Y = np.meshgrid(np.arange(0,num_tiles,1),np.arange(0,num_tiles,1))

    cached_energies[X,Y,2] = cached_energies[Y,X,0]

    cached_energies[X,Y,3] = cached_energies[Y,X,1]


    grid = np.arange(0,num_tiles,1,dtype=int).reshape((rows,columns))

    #print("cached") # debug
    
    return total_energy(grid,cached_energies)


@njit(parallel = False, fastmath = True) # cannot paralellize because of numpy advanced indexing of the array
def cache_energies(color, array, columns, rows, num_tiles, energyFunction):
    '''
    Now we disect each tile and save it's boarder vectors in a list of dictionaries.
    The  dictionaries will contain labels "top", "bottom", "left", "right"
    '''

    if not color:
        gray_matrix = array
        width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
        length = gray_matrix.shape[0]
        tile_width = width//columns
        tile_length = length//rows

        tiles = -np.ones(( num_tiles, 4, max(tile_length,tile_width)),dtype=np.float32)
        ''' first index controlls which tile we are looking at. The next index indicates the side of the tile 0-3 and the final dimension contains the vector associated with each side with and leftover space being filled with -1s '''

        for i in prange(num_tiles):
            m, n = divmod(i,columns)
            tiles[i,0,:tile_width] = gray_matrix[tile_length*m,tile_width*n:tile_width*(n+1)]
            tiles[i,2,:tile_width] = gray_matrix[tile_length*(m+1)-1,tile_width*n:tile_width*(n+1)]
            tiles[i,1,:tile_length] = gray_matrix[tile_length*m:tile_length*(m+1),tile_width*n]
            tiles[i,3,:tile_length] = gray_matrix[tile_length*m:tile_length*(m+1),tile_width*(n+1)-1]

        cached_energies = cache_energies_grayscale(num_tiles,energyFunction,tiles,tile_width,tile_length)
                    
    else:
        color_volume = array
        width = color_volume.shape[1] # horizontal distance - should be the shoter of the two
        length = color_volume.shape[0]
        tile_width = width//columns
        tile_length = length//rows

        tiles = -np.ones(( num_tiles, 4, max(tile_length,tile_width,3) ),dtype=np.float32)

        for i in prange(num_tiles):
            m, n = divmod(i,columns)
            tiles[i,0,:tile_width,:] = color_volume[tile_length*m,tile_width*n:tile_width*(n+1),:]
            tiles[i,2,:tile_width,:] = color_volume[tile_length*(m+1)-1,tile_width*n:tile_width*(n+1),:]
            tiles[i,1,:tile_length,:] = color_volume[tile_length*m:tile_length*(m+1),tile_width*n]
            tiles[i,3,:tile_length,:] = color_volume[tile_length*m:tile_length*(m+1),tile_width*(n+1)-1,:]

    return cached_energies


@njit(fastmath=True, parallel=True) # this function takes the bulk of the time so hopefully numba can speed it up; The energy doesn't need to be perfect since it has such large magnitude anyway. Thus we enable fastmath.
def cache_energies_grayscale(num_tiles:int,energyFunction:Callable[[np.ndarray,np.ndarray],np.float32],tiles:np.ndarray,tile_width:int, tile_length:int)->np.ndarray:
    cached_energies = np.zeros((num_tiles,num_tiles,4),dtype=np.float32)

    for i in prange(num_tiles): # run the first loop in parallel - innner loops apparently cannot be parallelized
        for j in range(num_tiles):
            for d_i in range(2):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cached_energies[i,j,d_i] = np.inf
                else:
                    if d_i == 0:
                        cached_energies[i,j,d_i] = energyFunction( tiles[i, d_i, :tile_width], tiles[j, (d_i + 2) % 4, :tile_width] ) # cutting off at tile_width shouldn't matter since they subtract out anyway, but in the case of some other energy function this is a good practice to do anyway
                    else: # d_i == 1
                        cached_energies[i,j,d_i] = energyFunction( tiles[i, d_i, :tile_length], tiles[j, (d_i + 2) % 4, :tile_length] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies


@njit(fastmath=True, parallel=True) # this function takes the bulk of the time so hopefully numba can speed it up; The energy doesn't need to be perfect since it has such large magnitude anyway. Thus we enable fastmath.
def cache_energies_color(num_tiles:int,energyFunction:Callable[[np.ndarray,np.ndarray],np.float32],tiles:np.ndarray,tile_width:int, tile_length:int)->np.ndarray:
    cached_energies = np.zeros((num_tiles,num_tiles,4),dtype=np.float32)

    for i in prange(num_tiles): # run the first loop in parallel - innner loops apparently cannot be parallelized
        for j in range(num_tiles):
            for d_i in range(2):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cached_energies[i,j,d_i] = np.inf
                else:
                    if d_i == 0:
                        cached_energies[i,j,d_i] = energyFunction( tiles[i, d_i, :tile_width], tiles[j, (d_i + 2) % 4, :tile_width,:] ) # cutting off at tile_width shouldn't matter since they subtract out anyway, but in the case of some other energy function this is a good practice to do anyway
                    else: # d_i == 1
                        cached_energies[i,j,d_i] = energyFunction( tiles[i, d_i, :tile_length], tiles[j, (d_i + 2) % 4, :tile_length,:] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies



# cannot use numba here due to advanced indexing
def total_energy(simGrid, cache_energies) -> float:

    # only rows 1..N-1 AND columns 1..N-1
    top_current = simGrid[1:, :]
    left_current = simGrid[:,1:]
    
    top_neighbors  = simGrid[:-1, :]
    left_neighbors = simGrid[:, :-1]

    energy_top = np.sum(cache_energies[top_current, top_neighbors, 0])
    energy_left = np.sum(cache_energies[left_current, left_neighbors, 1])

    return energy_top + energy_left



if __name__ == "__main__":

    ''' Compatability function'''
    from numpy.linalg import norm
    #compatability = lambda x,y: norm(x-y)/len(x)

    print(compute_energy(file = "Inputs/"+"test.jpg", color=False, energyFunction = compatability,puzzle_shape=(8,8)))
    print(compute_energy(file = "Inputs/"+"Original_Squirrel.jpg", color=True, energyFunction = compatability,puzzle_shape=(8,8)))
    print(compute_energy(file = "Inputs/"+"Original_RainbowFlower.jpg", color=True, energyFunction = compatability,puzzle_shape=(8,8)))
    #print(compute_energy(file = "Inputs/"+"Nebula_Puzzle.jpg", color=True, energyFunction = compatability,puzzle_shape=(40,60)))