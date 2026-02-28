'''
The goal of this document is to further refine the genetic and annealing methods by chaching compatability data.
'''

import numpy as np
from numpy.random import random, randint, choice
from time import sleep # debug tool
import cv2 # file I/O
from numba import prange, njit


'''
"Smart" Annealing
Here I try to adjust the annealing in a few ways;
a) replace my global energy recalculations in each markov step with local ones which should save compute time.
b) Add more annealing movement options that take into account the global energy information and try to move compatable tiles near each other.
c) Also try adding some more global rearangements to get blocks of aggregated tiles into the correct absolute position.
d) Create a more precise temperature schedule since the current geometric cooling seems to waste a lot of time spuddling on high temperatures.
'''

#
#------------------------------------------
#
# Simulated Annealing Class
#
#------------------------------------------
#

class simulation_grid:
    def __init__(self, grid, dict_list, cached_energies, T0=10., Tf=0.5, geometric_decay_rate=0.9999):
        self.simGrid = grid
        self.tile_data = dict_list
        self.grid_shape = self.simGrid.shape # note that this is the shape of the non-padded
        self.cached_energies = cached_energies # as far as I know, this just points to the original array so we are not actually making a copy of the large energy array
        self.energy = self.total_energy() # set the total energy on creating the class
        # for all intents and purposes recall that 0 = top, 1 = left, 2 = bottom, 3 = right which is quite a different ordering than the previous verison.
        #annealing constants:
        self.T0 = T0
        self.Tf = Tf
        self.geometric_rate = geometric_decay_rate #0.9999999 - takes a few hours #0.9999 - takes a few seconds

    def best_buddies(self):
        # identify the best-buddies - used for the genetric algorithm
        return
    
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
        array = array.astype(int) # we need to switch from uint8 since we are adding -1s.
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
        bottom_energy[boundaries[:,2]] = self.cached_energies[grid[rows[boundaries[:,2]]+1,columns[boundaries[:,2]]], current[boundaries[:,2]], 0] 
        right_energy[boundaries[:,3]] = self.cached_energies[grid[rows[boundaries[:,3]],columns[boundaries[:,3]]+1], current[boundaries[:,3]],1] 

        return top_energy + left_energy + bottom_energy + right_energy # total energy of local interactions

    
    def markovStep(self, temperature: float):
        '''
        Docstring for markovStep
        Mutates the self grid in accordance with the metropolis algorithm
        '''

        mode = randint(0,3) # choose from several difference mutation options at random

        '''Ideally at some point we make each move dependant on the tempurature schedule; we prefer larger moves at low T so that we can cement absolute positions
        though we might have to implement the genetic algorithm again to get the absolute positions, the idea in that method was decent'''

        new_grid, previous_contribution,new_contribution = self.perturb_grid(mode)

        energy_change = new_contribution - previous_contribution

        boltzmann_factor = np.exp(min(-energy_change,1.)/temperature) # min() prevent overlow in exp()

        if random() < boltzmann_factor: # always except whern previous >= new, sometimes accept an increase in the energy - dig out of local minima.
            self.simGrid = new_grid
            self.energy += energy_change

    def perturb_grid(self,mode):
        new_grid = np.copy(self.simGrid)
        if mode == 0 or mode == 2: # swap two points with a preference for swapping points with high local energies
            ''' We are changing this from the previous version to bias choosing points with high local energies
             We can use self.cached_energies to quickly lookup the local energy of a tile in O(1) time - just indexing an array at most four times
             There is still some small chance to swap two random rows'''
            Failed = False
            if random() < 0.95: # with 95% chance sample some pieces and choose the one with the highest local energy to swap with another point with high local energy
                # sample 20 pieces from the puzzle and compute their local energies
                sample_cols = randint(0,self.grid_shape[1],(20,1),dtype=int) # we don't turn off replacement, but the chance of duplicates is relatively low so it won't matter
                sample_rows = randint(0,self.grid_shape[0],(20,1),dtype=int)
                samples = np.hstack([sample_rows,sample_cols],dtype=int)
                worst_sample = samples[np.argmax(self.local_energy(self.simGrid,samples))] # Gets sample with the highest local energy; this return an index of a 2d array
                worst_sample = (worst_sample[0],worst_sample[1]) # it is important that this is a tuple, if it is a list or an array, then numpy indexing treats it as two separate indicies to be looked up
                # now we choose a direction and find its most optimal position in that direction and swap with the tile current in that position.
                d = randint(0,4)
                if d in {0,1}:
                    best_partner = np.argmin(self.cached_energies[self.simGrid[worst_sample],:,d]) # should return the index of the best partner in that direction
                else:
                    best_partner = np.argmin(self.cached_energies[:,self.simGrid[worst_sample],d-2])
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
                    worst = self.simGrid[worst_sample]
                    partner = self.simGrid[partner_location]
                    if worst == partner: # this happens with suprising frequency, it would nice to figure out why
                        Failed = True
                    else:
                        new_grid[partner_location] = worst
                        new_grid[worst_sample] = partner

            else: # with 5% chance swap two random pieces (more than 5% once you account for the failed swaps above)
                Failed = True

            if Failed: # make a random swap
                while True:
                    piece_rows = choice(list(range(self.grid_shape[0])),2,replace=True)
                    piece_cols = choice(list(range(self.grid_shape[1])),2,replace=True) # choosing the indicies for the points to swap
                    if piece_rows[0] != piece_rows[1] or piece_cols[0] != piece_cols[1]:
                        break # get out of loop when the points are unique

                worst = self.simGrid[piece_rows[0],piece_cols[0]]
                partner = self.simGrid[piece_rows[1],piece_cols[1]] # these names, unlike above, don't mean anything. Instead they are simply used for consistancy with the energy computations below

                new_grid[piece_rows[0],piece_cols[0]] = partner
                new_grid[piece_rows[1],piece_cols[1]] = worst

                
                partner_location = (piece_rows[1],piece_cols[1])
                worst_sample = (piece_rows[0],piece_cols[0])

            # compute the energy comtributions (only compute the local contributions)
            # we do this by recomputing all of the interactions of the swapped points and then taking the difference, if the two random points are adjacent, care must be taken not to double count the side on which the interact

            '''previous_contribution = self.local_energy(self.simGrid,np.array([[worst_sample[0],worst_sample[1]]])) + self.local_energy(self.simGrid,np.array([[partner_location[0],partner_location[1]]]))
            new_contribution = self.local_energy(new_grid,np.array([[worst_sample[0],worst_sample[1]]])) + self.local_energy(new_grid,np.array([[partner_location[0],partner_location[1]]]))
            previous_contribution -= (self.cached_energies[partner,worst,0] * ( worst == self.simGrid[partner_location[0]-(1 * (partner_location[0] != 0) ),partner_location[1]] )
                                    + self.cached_energies[partner,worst,2] * ( worst == self.simGrid[partner_location[0]+(1 * (partner_location[0] != self.grid_shape[0]-1) ),partner_location[1]] )
                                    + self.cached_energies[partner,worst,1] * (worst == self.simGrid[partner_location[0],partner_location[1]-(1 * (partner_location[1] != 0))] )
                                    + self.cached_energies[partner,worst,3] * (worst == self.simGrid[partner_location[0],partner_location[1]+(1 * (partner_location[1] != self.grid_shape[1]-1))] )
                                    ) # remove double counts if they are neighbors
            new_contribution -=  (self.cached_energies[partner,worst,0] * ( worst == new_grid[worst_sample[0]-(1 * ( worst_sample[0] != 0 )),worst_sample[1]] )
                                    + self.cached_energies[partner,worst,2] * ( worst == new_grid[worst_sample[0]+(1 * (worst_sample[0] != self.grid_shape[0]-1)),worst_sample[1]] )
                                    + self.cached_energies[partner,worst,1] * (worst == new_grid[worst_sample[0],worst_sample[1]-(1 * (worst_sample[1] != 0))] )
                                    + self.cached_energies[partner,worst,3] * (worst == new_grid[worst_sample[0],worst_sample[1]+(1 * (worst_sample[1] != self.grid_shape[1]-1))] )
                                    )''' # this is about 2x slower than just recomputing the grid energies - somehow
            previous_contribution = self.energy
            new_contribution = self.total_energy_grid(new_grid)
            
                
        elif False: # This serves a redundant function to the subarray swapping below and seems to be less effective
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

        elif mode == 1:
            ''' Here I try to move an entire block of tiles
            For simplicity we choose a rectangle shape and swap it with another rectangle with the same dimensions
            This allows us to keep pieces that are already matched together'''
            rows, cols = self.grid_shape

            while True:
                while True:
                    r = randint(1,rows+1) # must have at least one row
                    c = randint(1,cols+1)
                    if (r != 1 and c != 1) and not (r >= rows//2 and c >= cols//2): # can't be 1x1 because that is taken care of by mode 0 and 1; also cant have both r and c too large or the below check will fail so we just nip that in the bud here
                        break
                x1, y1 = ( randint(0,self.grid_shape[0]-r+1), randint(0,self.grid_shape[1]-c+1) )# top-left point of the first subarray; we choose this given r and c so that the subarray is not outside of the grid
                x2, y2 = ( randint(0,self.grid_shape[0]-r+1), randint(0,self.grid_shape[1]-c+1) ) # top-left point of the second subarray
                if not (
                    x1 + r <= x2 or x2 + r <= x1 or y1 + c <= y2 or y2 + c <= y1):
                    continue  # resample if the rectangles overlap
                break
            new_grid[x2:x2+r,y2:y2+c] = self.simGrid[x1:x1+r,y1:y1+c]
            new_grid[x1:x1+r,y1:y1+c] = self.simGrid[x2:x2+r,y2:y2+c]

            # now we compute the local change in energy; this is a bit more intensive than the single swaps above; for now I'll just recompute the enture grid energy
            previous_contribution = self.energy
            new_contribution = self.total_energy_grid(new_grid)

            # This (below) was my first attempt - I tired to cleverly choose the sub-arrays so that would not overlap - I would like to revisit this in the future but in the pursuit of functionality,
            # for now I'm just going to sample equally sized rectangles until we find two that don't overlap
            """# choose a tile to be the upper left of the rectangle:
            samples = (0,0)
            while samples == (0,0): # don't allow the top-left corner of the array since then we are unable to pick a complementary array
                sample_cols = randint(0,self.grid_shape[1],dtype=int)
                sample_rows = randint(0,self.grid_shape[0],dtype=int)
                samples = (sample_rows,sample_cols) # this is the point that we will
            d = choice([0,1]) # choose a direction; if top is chosen the partner subarray will be to the top of the subarray; similar for left; we need not consider bottom and right since we can make these swaps by choosing the would be partner tile as the starting tile as the the directions are reversed
            if samples[0]==0: # then d can only be left; I would like to write these as branchless but I can't find a concise way at the moment
                d = 1
            elif samples[1]==0: #then d can only be top
                d = 0
            if d == 0: # d is "top"
                '''the only condition on the column number is that we must be inside the grid so we need c < self.grid_shape[1] - samples[1]
                The rows have two conditions, first r < self.grid_shape[1] - samples[1] so that the rectangle stays within the grid
                but also that r < samples[1] so that the reciprical subarray has enough space to be fit above the present subarray'''
                r = randint(1,np.min([self.grid_shape[0]-(samples[0]+1),samples[0]])) # number of rows for the rectangle to have; must have at least 1 row
                c = randint(1,max(self.grid_shape[1]-samples[1],2)) # number of columns for the rectangle to have; must have at 
                '''Now we need to pick the upper-left corner of the partner subarray; it cannot overlap with the original sub-array so there are restrictions on its location
                for one, it cannot be placed within c columns of the right side of the array. As far as the rows are concerned the requirements are more delicate.
                However we will simply make the assumption that the difference betweent the x-indicies of the top left corners must be at least r. This need not be the case if 
                the sub array has few columns but for the sake of simplicity I do not consider that case.'''
                y2 = randint(0,self.grid_shape[1]-c+1) # note that here we can pick the point (0,0)
                x2 = randint(0,samples[0]-r+1)
            else: # d is 'left'
                '''Similar considerations to the rows apply'''
                r = randint(1,max(2,self.grid_shape[0]-samples[0]))
                c = randint(1,np.min([self.grid_shape[1]-samples[1],1+samples[1]]))
                x2 = randint(0,self.grid_shape[0]-r+1)
                y2 = randint(0,samples[1]-c+1)

            # make the swap
            new_grid = np.copy(self.simGrid)
            new_grid[x2:x2+r,y2:y2+c] = self.simGrid[samples[0]:samples[0]+r,samples[1]:samples[1]+c]
            new_grid[samples[0]:samples[0]+r,samples[1]:samples[1]+c] = self.simGrid[x2:x2+r,y2:y2+c]"""

        return new_grid, previous_contribution, new_contribution # eventually, return the energy changes once that is set up too.

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
            #print(f"Energy: {self.energy}, Temperature: {T}") # to track progress
            self.markovStep(T)
            T = self.cooling_schedule_geometric(self.T0,self.geometric_rate,i) #update the tempurature
            #T = self.cooling_schedule_optimal(self.T0,i) #update the tempurature
            '''if it takes 4 seconds to vomplete the geometric regieme it will take 1e3/36 hours to complete the logarithmic regieme accourding to my back of the envalope calculatiosn.'''
            i += 1

#
#
#------------------------------------------
#
# I/O Functions
#
#------------------------------------------
#


def generate_simGrid_from_file(filename="Inputs/Squirrel_Puzzle.jpg", grid_size=(8,8), color=True, energy_function = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y)), T0=10., Tf=0.5, geometric_decay_rate=0.9999 ) -> simulation_grid:
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
        
        tile_sides = -np.ones( ( num_tiles, 4, max(tile_length,tile_width) ), dtype=np.float32)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tiles.append(gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)]) # need this one to reconstruct the array later
                index = i * grid_size[0] + j
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

        tile_sides = -np.ones( ( num_tiles, 4, max(tile_length,tile_width), 3 ), dtype=np.float32)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tiles.append(color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:]) # need this one to reconstruct the array later
                index = i * grid_size[0] + j
                tile_sides[index, 0, :tile_width, :] = color_volume[tile_length*i,tile_width*j:tile_width*(j+1), :]
                tile_sides[index, 2, :tile_width, :] = color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1), :]
                tile_sides[index, 1, :tile_length, :] = color_volume[tile_length*i:tile_length*(i+1),tile_width*j, :]
                tile_sides[index, 3, :tile_length, :] = color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1, :]

        del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

        cached_energies = cache_energies_color(num_tiles, tile_sides, energy_function, tile_length,tile_width)

    grid = np.arange(0,num_tiles,1,dtype=int).reshape((grid_size[0],grid_size[1])) # the representation of the image; using uint8 because nothing is negative or bigger than 255 and thus using any other integer system would be wasteful


    tiles = np.array(tiles, dtype=object) # apparently you can make a list of arrays into an array - this makes indexing later much easier - this is a change from the previous version

    print("cached tile energies")
    

    return simulation_grid(grid, tiles, cached_energies, T0, Tf, geometric_decay_rate)


def annealing_reconstruct(simulation : simulation_grid, color = True):
    tile_width = (simulation.tile_data[0].shape[1]) # length of the top of an arbitrary tile
    tile_length = (simulation.tile_data[0].shape[0]) # length of the left of an arbitrary tile

    if color:

        width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
        length = tile_length * simulation.grid_shape[0]

        resotred_page = np.empty((length,width,3),dtype=np.uint8)

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.simGrid[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = simulation.tile_data[dict_index]
    else:

        width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
        length = tile_length * simulation.grid_shape[0] 

        resotred_page = np.empty((length,width),dtype=np.uint8)

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.simGrid[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = simulation.tile_data[dict_index]

    return resotred_page.astype(np.uint8) # jpg can only handle this resolution anyway

def save_annealing_output(filename, simulation : simulation_grid, color = True, reconstruction = None):
    if reconstruction is None:
        reconstruction = annealing_reconstruct(simulation,color)

    cv2.imwrite(filename, reconstruction)

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