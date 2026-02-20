'''
The goal of this document is to further refine the genetic and annealing methods by chaching compatability data.
'''

import numpy as np
from numpy.random import random, randint, choice
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

tiles = np.array(tiles, dtype=object) # apparently you can make a list of dictionaries into an array - this makes indexing later much easier - this is a change from the previous version

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
                energy += self.gradient_interact_energy(self.simGrid,(i,j))
        return energy
    
    def total_energy_grid(self,grid) -> float: # same as the above function except uses an arbitrary grid instead of self.simGrid; grid must have the same shape as self.simGrid 
        energy = 0.
        for i in range(1,self.grid_shape[0]): # skip firt row
            for j in range(1,self.grid_shape[1]): # skip first column
                # we don't want to double count interactions so we first only compute the energies to the left and obove each point (skipping the topmost and leftmost row/column)
                # then since the edges do not interact we can stop here since each interacting edge has been counted exactly once.
                energy += self.gradient_interact_energy(grid,(i,j))
        return energy
    
    def interaction_energy(self, grid, grid_point:tuple) -> float:
        '''grid_point is an index of the 2D array self.grid
        This function computes the interaction energy at the top and left sides of a tile; we only compute these two to prevent overcounting as we iterate through the array
        We modify this from the frist version in that now we simply need to search the self.cached_energies - this should make the function much faster since we only search an array instead
        of search several dictionaries and perform arithmetic.'''

        row = grid_point[0]
        column = grid_point[1]

        top_neighbor = self.tile_data[grid[row-1,column]]
        left_neighbor = self.tile_data[grid[row,column-1]]
        current = self.tile_data[grid[row,column]]

        return self.cached_energies[current,top_neighbor,0] + self.cached_energies[current,left_neighbor,1]
    
    def check_boundary(self, grid_points:np.ndarray) -> np.ndarray:
            """
            This function determines which sides of a specified grid_point are on the boundary of the grid.
            The previous version could only process a single coordinate pair at a time, I have changed this in the following way:
            now instead of inputting a tuple we input an Nx2 array where N is the number of points that we want to check.
            We then do logic on the array.
            Output is Nx4 array
            """
            rows = grid_points[:,0] # 1D array of all row coordinates
            cols = grid_points[:,1] # 1D array of all column coordinates
            shape = self.grid_shape

            top_open = rows > 0 # True when not bordering the top of the array  
            bottom_open = rows < self.grid_shape[0]-1
            left_open = cols > 0
            right_open = cols < self.grid_shape[1]-1
            '''This should be far more efficient than the code from before'''

            return np.vstack([top_open, left_open, bottom_open, right_open]).reshape((4,len(rows))).transpose() # this was a dictionary but not it is an Nx4 array where the second dimension takes the standard direction ordering thus far.
    
    def local_energy(self,grid_points:np.ndarray) -> float:
        '''Finds the energy of a single grid tile. Note that this is different from the interaction energy function above as this function returns
        the energy on each side and not just on the top and left sides
        The function takes in an Nx2 array of coordinates for N points. It returns a length N array'''
        rows = grid_points[:,0]
        columns = grid_points[:,1]

        boundaries = self.check_boundary(grid_points)
        current = self.tile_data[grid[rows,columns]]


        top_energy = self.cached_energies[self.tile_data[current,grid[row-1,column]],0] if boundaries[0] else 0. # only compute the energy on the top if there is a top neighbour
        left_energy = self.cached_energies[self.tile_data[current,grid[row,column-1]],1] if boundaries[1] else 0.
        bottom_energy = self.cached_energies[self.tile_data[current,grid[row+1,column]],2] if boundaries[2] else 0.
        right_energy = self.cached_energies[self.tile_data[current,grid[row,column+1]],3] if boundaries[3] else 0.

        return top_energy + left_energy + bottom_energy + right_energy # total energy of local interactions

    
    def markovStep(self, tempurature: float):
        '''
        Docstring for markovStep
        Mutates the self grid in accordance with the metropolis algorithm
        '''

        mode = randint(0,4) # choose from several difference mutation options at random

        if mode == 0 or mode == 3: # swap two points with a preference for swapping points with high local energies
            ''' We are changing this from the previous version to bias choosing points with high local energies
             We can use self.cached_energies to quickly lookup the local energy of a tile in O(1) time - just indexing an array at most four times
             There is still some small chance to swap two random rows'''
            if random() < 0.95: # with 95% chance sample some pieces and choose the one with the highest local energy to swap with another point with high local energy
                # sample 20 pieces from the puzzle and compute their local energies
            
            else: # with 5% chance swap two random pieces
                piece_rows = choice(list(range(self.grid_shape[0])),2,replace=False)
                piece_cols = choice(list(range(self.grid_shape[1])),2,replace=False) # choosing the indicies for the points to swap

                new_grid = np.copy(self.simGrid)
                new_grid[piece_rows[0],piece_cols[0]] = self.simGrid[piece_rows[1],piece_cols[1]]
                new_grid[piece_rows[1],piece_cols[1]] = self.simGrid[piece_rows[0],piece_cols[0]]

            # compute the energy comtributions (only compute the local contributions)

            #previous_contribution = 
            #new_contribution = 
                

        elif mode == 1:
            '''
            Same as in the previous version, use np.roll to permute columns or rows 
            The idea of this movement is that once we assemble chunck of related pieces, this function can move the chuncks into the correct absolute position.
            '''
            if random() < 0.5: # permute rows with 50% probability
                new_grid = np.roll(self.simGrid, randint(0,self.grid_shape[0]),axis=0)
            else: # permute columns
                new_grid = np.roll(self.simGrid, randint(0,self.grid_shape[1]),axis=1)

            #previous_contribution = self.energy
            #new_contribution = self.total_energy_grid(new_grid)
            # should just beable to recompute along the affected row or column

        elif mode == 2:
            ''' Here I try to move an entire block of tiles
            For simplicity we choose a rectangle shape and swap it with another rectangle with the same dimensions
            This allows us to keep pieces that are already matched together'''

        previous_contribution = self.energy # I do want to add these into the indevidual blocks; but for testing purposes and while the grids are small enough, I'll keep them here and recompute the entire grid energy
        new_contribution = self.total_energy_grid(new_grid)

        energy_change = previous_contribution - new_contribution

        boltzmann_factor = np.exp(-energy_change/tempurature)

        if random() < boltzmann_factor: # always except whern previous >= new, sometimes accept an increase in the energy - dig out of local minima.
            self.simGrid = new_grid
            self.energy += energy_change


image = simulation_grid(grid, tiles, cache_energies)