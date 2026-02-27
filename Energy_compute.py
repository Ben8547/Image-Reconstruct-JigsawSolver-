import numpy as np
import cv2
import warp as wp

wp.init()


def compute_energy(file,color=True, energyFunction=lambda x,y: lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y)),puzzle_shape = (8,8)):

    rows = puzzle_shape[0]
    columns = puzzle_shape[1]
    num_tiles = rows*columns

    grid = np.arange(0,num_tiles,1,dtype=np.uint8).reshape((rows,columns))

    '''Load in the test file (permanent)'''
    if color:
        color_volume = cv2.imread(file, cv2.IMREAD_COLOR)

        '''
        This outputs a 3 dimensional array: (hight, width, 3)
        We will need to store data differently taking this into account.
        '''
    else:
        gray_matrix = cv2.imread(file, cv2.IMREAD_GRAYSCALE)


    '''
    Now we disect each tile and save it's boarder vectors in a list of dictionaries.
    The  dictionaries will contain labels "top", "bottom", "left", "right"
    '''
    tiles = []

    if not color:
        width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
        length = gray_matrix.shape[0]
        tile_width = width//columns
        tile_length = length//rows
        #print(tile_length)

        for i in range(rows):
            for j in range(columns):
                tiles.append({
                    0: gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)],
                    2: gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)],
                    1: gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j],
                    3: gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1],
                    "entire": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] # need this last one to reconstruct the array later
                })
        del gray_matrix # no need to store a large matrix any longer than we need it. We only need the boarders anyway
    else:
        width = color_volume.shape[1] # horizontal distance - should be the shoter of the two
        length = color_volume.shape[0]
        tile_width = width//columns
        tile_length = length//rows
        #print(tile_length)

        for i in range(rows):
            for j in range(columns):
                tiles.append({
                    0: color_volume[tile_length*i,tile_width*j:tile_width*(j+1),:], # top
                    2: color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1),:], # bottom
                    1: color_volume[tile_length*i:tile_length*(i+1),tile_width*j,:], # left
                    3: color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1,:], # right
                    "entire": color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] # need this last one to reconstruct the array later
                }) # we use the wierd ordering so that we can use modular arithmatic to send top to bottom and left to right easily (and the reverse)

        del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

    grid = np.arange(0,num_tiles,1,dtype=np.uint8).reshape((rows,columns))

    cached_energies = cache_energies(num_tiles,energyFunction,tiles)

    #Since we only did top and left, we can recover bottom and right since the matrix has the following property cache[i,j,0] = cache[j,i,2] and cache[i,j,1] = cache[j,i,3]
    # by only computing half of the directions in the loop we should halve the compute time of the loop

    X, Y = np.meshgrid(np.arange(0,num_tiles,1,dtype=np.uint8),np.arange(0,num_tiles,1,dtype=np.uint8))

    cached_energies[X,Y,2] = cached_energies[Y,X,0]

    cached_energies[X,Y,3] = cached_energies[Y,X,1]

    #print("cached") # debug
    
    return total_energy(grid,cached_energies)

def cache_energies(num_tiles:int,energyFunction:callable,tiles:list[dict]):
    cached_energies = wp.zeros((num_tiles,num_tiles,4),dtype=float)

    for d in range(2):
        wp.launch(compute_single_energy,dim=num_tiles**2,inputs=[cached_energies,num_tiles,d])

    return cached_energies.numpy() # convert back to numpy array

@wp.kernel
def compute_single_energy(out, num_tiles,d):
    tid = wp.tid()
    i, j = divmod(tid,num_tiles)
    out[i,j,d] = 

def cache_energies_old(num_tiles:int,energyFunction:callable,tiles:list[dict])->np.ndarray:
    cached_energies = np.zeros((num_tiles,num_tiles,4),dtype=float)


    '''I've decided to use warp for caching the energies because for even moderaltely sized puzzles caching the energy takes several minutes and I don't believe
    that it can be vectorized.'''

    for i in range(num_tiles):
        #print(i) # debug
        for j in range(num_tiles):
            for d_i in range(2):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cached_energies[i,j,d_i] = np.inf
                else:
                    cached_energies[i,j,d_i] = energyFunction( tiles[i][d_i], tiles[j][(d_i + 2) % 4] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    return cached_energies


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
    compatability = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y))

    print(compute_energy(file = "Inputs/"+"Original_Nebula.jpg", color=True, energyFunction = compatability,puzzle_shape=(40,60)))
    #print(compute_energy(file = "Inputs/"+"Nebula_Puzzle.jpg", color=True, energyFunction = compatability,puzzle_shape=(40,60)))