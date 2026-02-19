import numpy as np
import cv2


def compute_energy(file,color=True, energyFunction=lambda x,y: np.linalg.norm(x-y)):

    grid = np.arange(0,64,1,dtype=np.int8).reshape((8,8))

    '''Load in the test file (permanent)'''
    if color:
        color_volume = cv2.imread(file, cv2.IMREAD_COLOR)
        #print(color_volume)
        #print(color_volume.shape)
        '''
        This outputs a 3 dimensional array: (hight, width, 3)
        We will need to store data differently taking this into account.
        '''
    else:
        gray_matrix = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        #print(gray_matrix)

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
                    "top": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j],
                    "bottom": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1],
                    "left": gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)],
                    "right": gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)],
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
                    "top": color_volume[tile_length*i:tile_length*(i+1),tile_width*j,:],
                    "bottom": color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1,:],
                    "left": color_volume[tile_length*i,tile_width*j:tile_width*(j+1),:],
                    "right": color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1),:],
                })

        del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway
    
    return total_energy(grid,tiles,energyFunction)


def total_energy(simGrid,tile_data,energyFunction):
        energy = 0.
        for i in range(1,simGrid.shape[0]): # skip firt row
            for j in range(1,simGrid.shape[1]): # skip first column
                # we don't want to double count interactions so we first only compute the energies to the left and obove each point (skipping the topmost and leftmost row/column)
                # then since the edges do not interact we can stop here since each interacting edge has been counted exactly once.
                energy += interaction_energy(simGrid,tile_data,(i,j),energyFunction)
        return energy

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
    #compatability = lambda x,y: norm(x-y)
    def compatability(x,y):
         return

    print(compute_energy(file = "Original_Squirrel.jpg", color=True, energyFunction = compatability))
    print(compute_energy(file = "Squirrel_Puzzle.jpg", color=True, energyFunction = compatability))
    print(compute_energy("test.jpg"))
    print(compute_energy("Original_RainbowFlower.jpg"))
    print(compute_energy("annealing-color.jpg"))