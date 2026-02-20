'''
The goal of this document is to further refine the genetic and annealing methods by chaching compatability data.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep # for debug utility

'''
Setup
'''

'''Deicde color or non-color'''

color = True

'''Load in the test file (permanent)'''

file = "test.jpg"

if color:
    color_volume = cv2.imread(file, cv2.IMREAD_COLOR)
    '''
    This outputs a 3 dimensional array: (hight, width, 3)
    We will need to store data differently taking this into account.
    '''
else:
    gray_matrix = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #print(gray_matrix)

'''Chop up the image'''

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
                "top": gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)],
                "bottom": gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)],
                "left": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j],
                "right": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1],
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

grid = np.arange(0,64,1,dtype=np.uint8).reshape((8,8)) # the representation of the image; using uint8 because nothing is negative or bigger than 255 and thus using any other integer system would be wasteful

''' Energy Function '''

compatability = lambda x,y: np.mean((x-y)**2) # energy function

'''
Cache every possible interaction energy
We will use the following data structure to store this information. Recall that each tile has an index from 0 to 63. Thus we define an array that is
64 x 64 x 4; We read the array as follows: define the list ['top', 'left', 'bottom', 'right'] so that we can correspond 0 to 'top', 1 to 'left' and so on.
Then the index [4,45,2] means that we are searching for the energy of the interaction between tile 5's bottom face and 46's top face. Similarly, [56,7,1] means that
we are looking up the energy between 57's left face and 8's right.

We do need to account for the diagonal elemets such as [3,3,2]. I'll set these to np.inf since I never want the tile to interact favorably with itself.

The compatability function can take vector inputs, but would condense them to a single value so I don't think that we can vectorize this; especially since we would need to
look everything up in the dictionary. Thus we'll cache with a loop.

We can also write to a file for easier lookup in the future if we run the same image multiple times in testing but adding this is not a highpriority.

Note that if we define sigma(i) = (i+2)%4 then 0->2 (top->bottom),  1->3, 2->0 and 3->1 exactly as desired to get the opposites; thus we don't even need the conversion dictionary that I
was using in my previous versions.
'''

cache_energies = np.zeros((64,64,4),dtype=float)

for i in range(64):
    for j in range(64):
        for d_i in range(4):
            if i == j: # diagonal elements are set to infinite since they can never happen anyway
                cache_energies[i,j,d_i] = np.inf
            else:
                cache_energies[i,j,d_i] = compatability( tiles[i][d_i], tiles[j][(d_i + 2) % 4] )

# this is actually suprisingly quick to compute though probably won't scale well for large puzzles. Luckily we only care about 64x64 right now.
# in the current case it might actually take longer to open a read a file than just recompute all of the energies
print("cached tile energies")

'''
Compute Best-Buddies
'''


'''
"Smart" Annealing
Here I try to adjust the annealing in a few ways;
a) replace my global energy recalculations in each markov step with local ones which should save compute time.
b) Add more annealing movement options that take into account the global energy information and try to move compatable tiles near each other.
c) Also try adding some more global rearangements to get blocks of aggregated tiles into the correct absolute position.
d) Create a more precise tempurature schedule since the current geometrix cooling seems to waste a lot of time spuddling on high tempuratures.
'''

class simulation_grid:
    def __init__(self, grid, dict_list, cached_energies):
        self.simGrid = grid
        self.tile_data = dict_list
        self.grid_shape = self.simGrid.shape
        self.cached_energies = self.cached_energes # as far as I know, this just points to the original array so we are not actually making a copy of the large energy array
        self.energy = self.total_energy() # set the total energy on creating the class
        # for all intents and purposes recall that 0 = top, 1 = left, 2 = bottom, 3 = right which is quite a different ordering than the previous verison.
    
    def total_energy(self) -> float:
        energy = 0.
        for i in range(1,self.grid_shape[0]): # skip firt row
            for j in range(1,self.grid_shape[1]): # skip first column
                # we don't want to double count interactions so we first only compute the energies to the left and obove each point (skipping the topmost and leftmost row/column)
                # then since the edges do not interact we can stop here since each interacting edge has been counted exactly once.
                energy += self.gradient_interact_energy((i,j))
        return energy
    
    def interaction_energy(self, grid_point:tuple) -> float:
        '''grid_point is an index of the 2D array self.grid
        This function computes the interaction energy at the top and left sides of a tile; we only compute these two to prevent overcounting as we iterate through the array
        We modify this from the frist version in that now we simply need to search the self.cached_energies - this should make the function much faster since we only search an array instead
        of search several dictionaries and perform arithmetic.'''

        row = grid_point[0]
        column = grid_point[1]

        top_neighbor = self.tile_data[self.simGrid[row-1,column]]
        left_neighbor = self.tile_data[self.simGrid[row,column-1]]
        current = self.tile_data[self.simGrid[row,column]]

        return self.cached_energies[current,top_neighbor,0] + self.cached_energies[current,left_neighbor,1]
    
    



image = simulation_grid(grid, tiles, cache_energies)