'''
The goal of this document is to further refine the genetic and annealing methods by chaching compatability data.
'''

import numpy as np
from numpy.random import random, randint, choice
import matplotlib.pyplot as plt
import cv2
from time import sleep # for debug utility


'''
"Smart" Annealing
Here I try to adjust the annealing in a few ways;
a) replace my global energy recalculations in each markov step with local ones which should save compute time.
b) Add more annealing movement options that take into account the global energy information and try to move compatable tiles near each other.
c) Also try adding some more global rearangements to get blocks of aggregated tiles into the correct absolute position.
d) Create a more precise temperature schedule since the current geometric cooling seems to waste a lot of time spuddling on high temperatures.
'''

class simulation_grid:
    def __init__(self, grid, dict_list, cached_energies):
        self.simGrid = grid
        self.tile_data = dict_list
        self.grid_shape = self.simGrid.shape # note that this is the shape of the non-padded
        self.cached_energies = cached_energies # as far as I know, this just points to the original array so we are not actually making a copy of the large energy array
        self.energy = self.total_energy() # set the total energy on creating the class
        # for all intents and purposes recall that 0 = top, 1 = left, 2 = bottom, 3 = right which is quite a different ordering than the previous verison.
        #annealing constants:
        self.T0 = 10.
        self.Tf = 0.5
        self.geometric_rate = 0.9999
    
    def total_energy(self) -> float:
        '''This should be much faster since I only use no explicit python loops instead of the 2 from the previous version
        Use the non-padded array: we don't need to do anything fancy with the edges so this is just easier'''
        top_current = self.simGrid[1:, :]
        left_current = self.simGrid[:,1:]
    
        top_neighbors  = self.simGrid[:-1, :] # returns the matrix from the topmost row the second to penultimate row; these will all have their bottoms measured
        left_neighbors = self.simGrid[:, :-1]

        energy_top = np.sum(self.cached_energies[top_current, top_neighbors, 0])
        energy_left = np.sum(self.cached_energies[left_current, left_neighbors, 1])

        return energy_top + energy_left
    
    def total_energy_grid(self,grid) -> float: # same as the above function except uses an arbitrary grid instead of self.simGrid; grid must have the same shape as self.simGrid 
        top_current = grid[1:, :]
        left_current = grid[:,1:]
    
        top_neighbors  = grid[:-1, :] # returns the matrix from the topmost row the second to penultimate row; these will all have their bottoms measured
        left_neighbors = grid[:, :-1]

        energy_top = np.sum(self.cached_energies[top_current, top_neighbors, 0])
        energy_left = np.sum(self.cached_energies[left_current, left_neighbors, 1])

        return energy_top + energy_left
    
    def check_boundary(self, grid_points:np.ndarray) -> np.ndarray:
            """
            This function determines which sides of a specified grid_point are on the boundary of the grid. In input points should be from the unpadded array.
            The previous version could only process a single coordinate pair at a time, I have changed this in the following way:
            now instead of inputting a tuple we input an Nx2 array where N is the number of points that we want to check.
            We then do logic on the array.
            Output is Nx4 array
            """
            rows = grid_points[:,0] # 1D array of all row coordinates
            cols = grid_points[:,1] # 1D array of all column coordinates

            top_open = rows > 0 # True when not bordering the top of the array  
            bottom_open = rows < self.grid_shape[0]-1
            left_open = cols > 0
            right_open = cols < self.grid_shape[1]-1
            '''This should be far more efficient than the code from before'''

            return np.vstack([top_open, left_open, bottom_open, right_open]).reshape((4,len(rows))).transpose() # this was a dictionary but not it is an Nx4 array where the second dimension takes the standard direction ordering thus far.
    
    def pad(self, array:np.ndarray) -> np.ndarray: # this is important for vectorizing many of the operations in this class but sees no direct use outside of the initialization of the class
        '''Adds a padding of -1 along each exposed face of a 2D array'''
        '''This is depreciated; we don't actually need this'''
        shape = array.shape
        # add top and bottom
        array = array.astype(np.int8) # we need to switch from uint8 since we are adding -1s.
        array = np.vstack([-np.ones((1,shape[1]),dtype=np.int8), array])
        array = np.vstack([array, -np.ones((1,shape[1]),dtype=np.int8)])
        # add left/right
        array = np.hstack([-np.ones((shape[0]+2,1),dtype=np.int8), array])
        array = np.hstack([array, -np.ones((shape[0]+2,1),dtype=np.int8)])
        return array
    
    def local_energy(self,grid,grid_points:np.ndarray) -> float:
        '''Finds the energy of a single grid tile. Note that this is different from the interaction energy function above as this function returns
        the energy on each side and not just on the top and left sides
        The function takes in an Nx2 array of coordinates for N points. It returns a length N array
        We use the padded array here to allow indexing withough if statements. We remove the -1s by musltiplying them by False
        Note that the input grid_points should be for the unpadded array - we convert to the padded indicies herein, the input grid should also be unpadded'''
        # ignore all the stuff about padding; we don't use padding anymore; instead I use logical masks
        rows = grid_points[:,0]
        columns = grid_points[:,1]
        # adding one account for the extra later of padding

        boundaries = self.check_boundary(grid_points) # Nx4 array
        current = grid[rows,columns] # array

        # I'm glad I watched that video on branchless code in C the other day, it is quite applicable and I probably would not have though to do this otherwise
        top_energy = np.zeros_like(current,dtype=np.float32)
        left_energy = np.copy(top_energy)
        right_energy = np.copy(top_energy)
        bottom_energy = np.copy(top_energy)

        top_energy[boundaries[:,0]] = self.cached_energies[current[boundaries[:,0]],grid[rows[boundaries[:,0]]-1,columns[boundaries[:,0]]],0]
        left_energy[boundaries[:,1]] = self.cached_energies[current[boundaries[:,1]],grid[rows[boundaries[:,1]],columns[boundaries[:,1]]-1],1] 
        bottom_energy[boundaries[:,2]] = self.cached_energies[current[boundaries[:,2]],grid[rows[boundaries[:,2]]+1,columns[boundaries[:,2]]],2] 
        right_energy[boundaries[:,3]] = self.cached_energies[current[boundaries[:,3]],grid[rows[boundaries[:,3]],columns[boundaries[:,3]]+1],3] 

        return top_energy + left_energy + bottom_energy + right_energy # total energy of local interactions

    
    def markovStep(self, temperature: float):
        '''
        Docstring for markovStep
        Mutates the self grid in accordance with the metropolis algorithm
        '''

        mode = randint(0,4) # choose from several difference mutation options at random

        '''Ideally at some point we make each move dependant on the tempurature schedule; we prefer larger moves at low T so that we can cement absolute positions
        though we might have to implement the genetic algorithm again to get the absolute positions, the idea in that method was decent'''

        if mode == 0 or mode == 3 or mode == 2: # swap two points with a preference for swapping points with high local energies
            ''' We are changing this from the previous version to bias choosing points with high local energies
             We can use self.cached_energies to quickly lookup the local energy of a tile in O(1) time - just indexing an array at most four times
             There is still some small chance to swap two random rows'''
            Failed = False
            if random() < 0.95: # with 95% chance sample some pieces and choose the one with the highest local energy to swap with another point with high local energy
                # sample 20 pieces from the puzzle and compute their local energies
                sample_cols = randint(0,self.grid_shape[1],(20,1),dtype=np.uint8) # we don't turn off replacement, but the chance of duplicates is relatively low so it won't matter
                sample_rows = randint(0,self.grid_shape[0],(20,1),dtype=np.uint8)
                samples = np.hstack([sample_rows,sample_cols],dtype=np.uint8)
                worst_sample = samples[np.argmax(self.local_energy(self.simGrid,samples))] # Gets sample with the highest local energy; this return an index of a 2d array
                worst_sample = (worst_sample[0],worst_sample[1]) # it is important that this is a tuple, if it is a list or an array, then numpy indexing treats it as two separate indicies to be looked up
                # now we choose a direction and find its most optimal position in that direction and swap with the tile current in that position.
                d = randint(0,4)
                best_partner = np.argmin(self.cached_energies[self.simGrid[worst_sample],:,d]) # should return the index of the best partner in that direction
                # make the swap:
                partner_location = np.argwhere( self.simGrid == best_partner )[0] # returns the matching piece, not the location to move the piece to; returns a 1x2 array which is why we must extract [0]
                partner_location[0] += -(d - 1) * ( d%2 == 0 )  # add to the indicies depending on direction. If d=0 we change by +1, if d = 2 we change by -1 else by 0. Thus we can take d-1 but only if d is even. 
                # We invert from the standard because we looked it up so that worst_sample was paired at it's d face, so we actaully need the complementary direction on the partner.
                partner_location[1] += -(d - 2) * (d%2 == 1) # need +1 if d = 1 and -1 if d = 3 else 0. d-2 if odd should work; I'm am quite proud of myself for the branchless logic today.
                partner_location = (partner_location[0],partner_location[1])
                # we still have not accounted for the piece being on the boarder
                if partner_location[0] < 0 or partner_location[0] > self.grid_shape[0]-1 or partner_location[1] < 0 or partner_location[1] > self.grid_shape[1]-1:
                    Failed = True # Then the swap is not valid so we move to a random swap instead
                else: # then we proceed with the swap
                    new_grid = np.copy(self.simGrid)
                    new_grid[partner_location] = self.simGrid[worst_sample]
                    new_grid[worst_sample] = self.simGrid[partner_location]
            else: # with 5% chance swap two random pieces (more than 5% once you account for the failed swaps above)
                Failed = True

            if Failed: # make a random swap
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
            # choose a tile to be the upper left of the rectangle:
            sample_cols = randint(0,self.grid_shape[1],dtype=np.uint8)
            sample_rows = randint(0,self.grid_shape[0],dtype=np.uint8)
            samples = (sample_rows,sample_cols)
            # choose the dimensions of the rectangle
            width = randint(0,self.grid_shape[0]-samples[0]) # num rows -1
            length = randint(0,self.grid_shape[1]-samples[1]) # num columns -1
            # this should always be contained within the simulation grid

            # choose the grid to swap with by choosing the upper left point
            sample_cols = randint(0,self.grid_shape[1]-width,dtype=np.uint8)
            sample_rows = randint(0,self.grid_shape[0]-length,dtype=np.uint8)
            partner_location = (sample_rows,sample_cols)

            # make the swap
            new_grid = np.copy(self.simGrid)
            new_grid[partner_location[0]:partner_location[0]+width+1,partner_location[1]:partner_location[1]+length+1] = self.simGrid[samples[0]:samples[0]+width+1,samples[1]:samples[1]+length+1]
            new_grid[samples[0]:samples[0]+width+1,samples[1]:samples[1]+length+1] = self.simGrid[partner_location[0]:partner_location[0]+width+1,partner_location[1]:partner_location[1]+length+1]

        previous_contribution = self.energy # I do want to add these into the indevidual blocks; but for testing purposes and while the grids are small enough, I'll keep them here and recompute the entire grid energy
        new_contribution = self.total_energy_grid(new_grid)

        energy_change = new_contribution - previous_contribution

        boltzmann_factor = np.exp(-energy_change/temperature)

        if random() < boltzmann_factor: # always except whern previous >= new, sometimes accept an increase in the energy - dig out of local minima.
            self.simGrid = new_grid
            self.energy += energy_change

    def cooling_schedule_optimal(self,T_0, k):
        '''Note that to reach a final temperature of T_f we
        need to run the algorithm for exp(T_0/T_f) and since T_f is small and T_0 large, this is usually a very large number.'''
        # only call if k > 1
        # The proof of Hajek, 1998 gaurentees that this cooling schedule leads to converge if T_0 > largest minimal depth of an element of the solution space
        # because of the logarithmic scaling, this scedule takes a long time to resolve so I may change it later.
        return T_0/np.log(k)
    
    def cooling_schedule_geometric(self,T_0,rate,k):
        return T_0 * (rate**k)


    def anneal(self):
        '''This method runs the annealing process by iternating through markov with a tempurature schedule'''
        T = self.T0
        i = 2
        while T > self.Tf: 
            self.markovStep(T)
            #print(f"Energy: {self.energy}, Temperature: {T}") # to track progress
            T = self.cooling_schedule_geometric(self.T0,self.geometric_rate,i) #update the tempurature
            #T = self.cooling_schedule_optimal(self.T0,i) #update the tempurature
            i += 1


if __name__ == "__main__": # so that we can just import the class if desired

    '''
    Setup
    '''

    '''Deicde color or non-color'''

    color = True

    '''Load in the test file (permanent)'''

    file = "RainbowFlower_Puzzle.jpg"

    if color:
        color_volume = cv2.imread("Inputs/"+file, cv2.IMREAD_COLOR)
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
                    cache_energies[i,j,d_i] = compatability( tiles[i][d_i], tiles[j][(d_i + 2) % 4] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    # this is actually suprisingly quick to compute though probably won't scale well for large puzzles. Luckily we only care about 64x64 right now.
    # in the current case it might actually take longer to open a read a file than just recompute all of the energies
    print("cached tile energies")

    '''
    Compute Best-Buddies
    '''


    simulation = simulation_grid(grid, tiles, cache_energies)
    print(f"Initial Energy: {simulation.energy}")
    simulation.anneal()

    '''Now that we have the ordered array, all that remains is to put the grayscale map back together.'''

    if color:
        resotred_page = np.zeros((length,width,3))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.simGrid[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = simulation.tile_data[dict_index]["entire"]

        resotred_page = resotred_page.astype(np.uint8) # jpg can only handle this resolution anyway
        cv2.imwrite("Outputs/"+f"annealing-color.jpg", resotred_page)
    else:
        resotred_page = np.zeros((length,width))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.simGrid[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = simulation.tile_data[dict_index]["entire"]
        resotred_page = resotred_page.astype(np.uint8)
        cv2.imwrite("Outputs/"+f"annealing-grayscale.jpg", resotred_page)

    print(f"Final energy {simulation.energy}")

    plt.imshow(resotred_page)
    plt.show()