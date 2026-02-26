import numpy as np
import cv2


def compute_energy(file,color=True, energyFunction=lambda x,y: lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y))):

    grid = np.arange(0,64,1,dtype=np.uint8).reshape((8,8))

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
        tile_width = width//8
        tile_length = length//8
        #print(tile_length)

        for i in range(8):
            for j in range(8):
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
        tile_width = width//8
        tile_length = length//8
        #print(tile_length)

        for i in range(8):
            for j in range(8):
                tiles.append({
                    0: color_volume[tile_length*i,tile_width*j:tile_width*(j+1),:], # top
                    2: color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1),:], # bottom
                    1: color_volume[tile_length*i:tile_length*(i+1),tile_width*j,:], # left
                    3: color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1,:], # right
                    "entire": color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] # need this last one to reconstruct the array later
                }) # we use the wierd ordering so that we can use modular arithmatic to send top to bottom and left to right easily (and the reverse)

        del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

    grid = np.arange(0,64,1,dtype=np.uint8).reshape((8,8))

    cache_energies = np.zeros((64,64,4),dtype=float)

    for i in range(64):
        for j in range(64):
            for d_i in range(2):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cache_energies[i,j,d_i] = np.inf
                else:
                    cache_energies[i,j,d_i] = energyFunction( tiles[i][d_i], tiles[j][(d_i + 2) % 4] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    #Since we only did top and left, we can recover bottom and right since the matrix has the following property cache[i,j,0] = cache[j,i,2] and cache[i,j,1] = cache[j,i,3]
    # by only computing half of the directions in the loop we should halve the compute time of the loop

    X, Y = np.meshgrid(np.arange(0,64,1,dtype=np.uint8),np.arange(0,64,1,dtype=np.uint8))

    cache_energies[X,Y,2] = cache_energies[Y,X,0]

    cache_energies[X,Y,3] = cache_energies[Y,X,1]
    
    return total_energy(grid,cache_energies)


def total_energy(simGrid, cache_energies) -> float:

    # only rows 1..N-1 AND columns 1..N-1
    top_current = simGrid[1:, :]
    left_current = simGrid[:,1:]
    
    top_neighbors  = simGrid[:-1, :]
    left_neighbors = simGrid[:, :-1]

    energy_top = np.sum(cache_energies[top_current, top_neighbors, 0])
    energy_left = np.sum(cache_energies[left_current, left_neighbors, 1])

    return energy_top + energy_left

'''def total_energy(simGrid, cache):
    energy = 0.0
    for i in range(1, simGrid.shape[0]):
        for j in range(1, simGrid.shape[1]):
            c = simGrid[i,j]
            energy += cache[c, simGrid[i-1,j], 0]
            energy += cache[c, simGrid[i,j-1], 1]
    return energy'''

def interaction_energy(simGrid, tile_data, grid_point:tuple, energyFunction) -> float:
        '''
        Docstring for interaction_energy
        
        :param grid_point: the index of element of the grid we are looking at; the index should be the [row,column] index form (not the value stored in the grid at that point which is the index of the list of dicts)
        :type grid_point: tuple
        :return: The interaction energy between the tile and its grid neighbors
        :rtype: float
        '''

        row = grid_point[0]
        column = grid_point[1]

        # for why we only count the top and right neighbors, see the above function "total_energy" - in short it precents double counting

        top_neighbor = tile_data[simGrid[row-1,column]]
        #bottom_neighbor = tile_data[sim_grid]
        left_neighbor = tile_data[simGrid[row,column-1]]
        #right_neighbor = tile_data[simGrid[row,column+1]]
        current = tile_data[simGrid[row,column]]

        # top side interaction
        top_energy = energyFunction(top_neighbor['bottom'], current['top'])
        # borrom side interaction
        #bottom_energy = norm(bottom_neighbor['top'] - current['bottom'])
        # top side interaction
        left_energy = energyFunction(left_neighbor['right'], current['left'])
        # top side interaction
        #right_energy = norm(right_neighbor['left'] - current['right'])

        return top_energy + left_energy# + bottom_energy + right_energy




if __name__ == "__main__":

    ''' Compatability function'''
    from numpy.linalg import norm
    #compatability = lambda x,y: norm(x-y)/len(x)
    compatability = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y))

    print(compute_energy(file = "Inputs/"+"Original_Squirrel.jpg", color=True, energyFunction = compatability))
    print(compute_energy(file = "Inputs/"+"Original_RainbowFlower.jpg", color=True, energyFunction = compatability))
    print(compute_energy(file = "Inputs/"+"Squirrel_Puzzle.jpg", color=True, energyFunction = compatability))
    print(compute_energy("ReadMeImages/"+"pure_annealed_squirrel_rolls_only.jpg",True,compatability))
    print(compute_energy(file = "Outputs/"+"annealing-color.jpg", color=True, energyFunction = compatability))
    print(compute_energy(file = "Outputs/"+"genome-color.jpg", color=True, energyFunction = compatability))