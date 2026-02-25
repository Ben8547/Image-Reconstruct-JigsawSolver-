import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep
from Annealing_Class import simulation_grid

#
#
#------------------------------------------
#
# Genome Class
#
#------------------------------------------
#

class Genome:
    '''
    This class runs the details of the genetic algorithm
    '''
    def __init__(self, grid_list, dict_list, cached_energies, numberGenerations:int = 100, parentsPerGeneration:int = 4, populationSize:int = 1000, T0:float=10., Tf:float=0.5, geometric_decay_rate:float=0.9999, updates=False):
        if len(grid_list) > populationSize:
            raise ValueError("grid_list cannot have more elements than populationSize")
        self.grid_shape = grid_list[0].shape # the seed grid - usually just the input puzzle image
        self.tile_data = dict_list # see the simulation_grid class
        self.cached_energies = cached_energies # see the simulation_grid class

        self.numberGenerations = numberGenerations # number of generations to run in the simulation
        self.parentsPerGen = parentsPerGeneration  # number of indeviduals to keep from the old generation at the begining of each new generation
        self.populationSize = populationSize # number of individuals in each generation
        self.chromosomes = np.empty(self.populationSize,dtype=object) # array of chromosome arrays; a chromosome is a permutation of the initial grid (a member of solution space  )
        self.chromosomes[0:len(grid_list)] = grid_list # populate the chromosomes with any initial chromosomes
        self.product = self.chromosomes[0] # at the end of simulation, this will contain the output
        self.energy = self.total_energy_grid(self.product)  # will contain the energy at the end of the simulation

        self.mutation_probability = 1./1000. # probability of random mutation when creating a child
        self.T0 = T0 # initial annealing parameter
        self.Tf = Tf # initial annealing parameter
        self.cooling_rate = geometric_decay_rate # initial annealing parameter

        self.updates = updates # boolean that determines weather to print updates after each generation completes

        self.directions = {0,1,2,3}

    def run_simulation(self):
        self.initial_anneal()
        for generation in range(self.numberGenerations):
            self.chromosomes = self.n_most_fit(self.parentsPerGen)
            self.energy = self.total_energy_grid(self.chromosomes[0])
            if self.updates:
                print(f"Best starting energy of generation {generation} is {self.energy}")
            for i in range(self.populationSize - self.parentsPerGen):
                parents  = np.random.choice(self.chromosomes[:self.parentsPerGen],2,replace=False)
                self.chromosomes[self.parentsPerGen + i] = self.generate_child(parent1=parents[0],parent2=parents[1])

        self.n_most_fit(1) # run this just to order the chromosome
        self.chromosomes[1:] = np.empty(shape=self.populationSize-1,dtype=object)
        self.initial_anneal() # the annealing won't disrupt a fully formed image, but if we are missing a few tile, the annealing might be able to fix it
        self.energy = self.total_energy_grid(self.chromosomes[0])
        self.product = self.chromosomes[0]

    def initial_anneal(self):
        ''' The genome class will perform well if matches have already been found, but our current algorithm is not great at finding said matches. Thus,
         We run annealing on the seed grids before the genetic algorithm in order to find these initial pairings; ideally there are repeats between parents '''
        for i in range(self.populationSize):
            if self.chromosomes[i] != None:
                annealed_chromosome = simulation_grid(self.chromosomes[i],self.tile_data,self.cached_energies,self.T0, self.Tf,self.cooling_rate)
                annealed_chromosome.anneal()
                self.chromosomes[i] = annealed_chromosome.simGrid
    
    def total_energy_grid(self,grid) -> float: 
        top_current = grid[1:, :]
        left_current = grid[:,1:]
    
        top_neighbors  = grid[:-1, :] # returns the matrix from the topmost row the second to penultimate row; these will all have their bottoms measured
        left_neighbors = grid[:, :-1]

        energy_top = np.sum(self.cached_energies[top_current, top_neighbors, 0])
        energy_left = np.sum(self.cached_energies[left_current, left_neighbors, 1])

        return energy_top + energy_left
    
    def n_most_fit(self,n:int):
        if n > self.populationSize:
            raise(ValueError)
        
        list_of_fitnesses = []
        for i in range(self.populationSize):
            if self.chromosomes[i] == None:
                list_of_fitnesses.append(np.inf) # highest energy, worst fitness
            else:
                list_of_fitnesses.append(self.total_energy_grid(self.chromosomes[i]))
        
        self.chromosomes = self.chromosomes[np.argsort(list_of_fitnesses)] # order the chromosomes from most fit to least fit (least to highest energy)

        return self.chromosomes[:n] # return the first n items of the list; will fail if n > length of chromosomes hence the earlier error message
    
    def generate_child(self,parent1, parent2):
        grid = -np.ones((1,1),dtype=int) # begin with an empty 1x1 array
        '''seed the child with the first tile.'''
        parent_adjacencies_lookup, parent_adjacencies = self.search_parents_for_same_adjacencies(parent1, parent2)
        if parent_adjacencies: # empty lists give false; lists with elements yield true
                grid[0,0] = np.random.choice(parent_adjacencies)
        else: # no adjacencies
            grid[0,0] = np.random.choice(parent1.ravel())

        used_tiles = [None, grid[0,0]] # add the used tile to the list - this cannot be added again
        unused_tiles = np.array([i for i in range(self.grid_shape[0]*self.grid_shape[1])],dtype=object) # this one specifically is an array because we need to use a mask on it later
        unused_tiles[grid[0,0]] = None
        num_used_tiles = 1

        while num_used_tiles < self.grid_shape[0]*self.grid_shape[1]:
            # start by searching for an element in the child in parent_adjacencies
            current_shape = grid.shape
            place_random = False
            if parent_adjacencies: # check that there even are duplicate adjacencies
                dual_adjacent_in_child = []
                for i, element in enumerate(grid.ravel()):
                    # i is the index of the raveled array; element is the integer representing the tile
                    m = i // current_shape[1] # row index of element
                    n = i % current_shape[1] # column index of element
                    if element in parent_adjacencies:
                        open = self.check_open_side(grid, m, n, current_shape)
                        repetative_neighbors_dict = parent_adjacencies_lookup[element] # should return a dictionary with the parental repeated neighbors on each side listed
                        if any( ( (repetative_neighbors_dict[d] not in used_tiles) and ( open[d] != False ) ) for d in self.directions ): # check that the intended neighbor has not already been place in the child; recall that None is in the list of used tiles so we get True if there is no ajacency
                            dual_adjacent_in_child.append(i) # this will be a list of all of the elements in the child that have repeated adjacencies in the parents that also have open sides in the grid
                        else: # then the partner of this tile is used already - there is no reason to consider either or them in future loops
                            parent_adjacencies.remove(element) # no reason to consider this element again on future loops
                if dual_adjacent_in_child: # check the list is non-empty; i.e. were any of the neighbors not already chosen?
                    # choose a tile which a neighbour
                    tile = np.random.choice(dual_adjacent_in_child) # tile index in the flat array
                    m = tile // current_shape[1] # row of tile
                    n = tile % current_shape[1] # column of tile
                    tile = grid[m,n]
                    # now find its common neighbours in the parents, and choose one at random
                    repetative_neighbors_dict = parent_adjacencies_lookup[tile]
                    valid_directions = []
                    for direction in self.directions:
                        if (repetative_neighbors_dict[direction] != None) and self.check_open_side(grid, m,n,current_shape)[direction]: # requires the element to be open in that direction and for it to actully have a valid neighbor
                            valid_directions.append(direction)
                    direction = np.random.choice(valid_directions) # choose a random valid direction

                    if np.random.random() < self.mutation_probability: # random mutation
                        placement = np.random.choice(unused_tiles[unused_tiles != None])
                    else:
                        placement = repetative_neighbors_dict[direction]
                    # place the tile in the chosen direction
                    grid, num_used_tiles = self.place_tile(grid,direction,placement,used_tiles, unused_tiles, current_shape, num_used_tiles, m, n)
                    
                else: # there are no dual adjacencies present in the child
                    place_random = True

            else: #there are no dulpicate adjacencies 
                place_random = True

            if place_random:
                # find elements that have open side:
                valid_elements = []
                for i, element in enumerate(grid.ravel()):
                    if element != -1:
                        # i is the index of the raveled array; element is the integer representing the tile
                        m = i // current_shape[1] # row index of element
                        n = i % current_shape[1] # column index of element
                        open = self.check_open_side(grid, m,n, current_shape)
                        any_open = any(open[j] for j in self.directions)
                        if any_open:
                            valid_elements.append(i)
                element_index = np.random.choice(valid_elements) # choose one of the valid elements to add on to
                m = element_index // current_shape[1]
                n = element_index % current_shape[1]
                element = grid[m,n]
                # now find the valid neighbors of that element
                valid_directions = []
                for direction in self.directions:
                        if self.check_open_side(grid, m, n, current_shape)[direction]: # requires the element to be open in that direction and for it to actully have a valid neighbor
                            valid_directions.append(direction)
                direction = np.random.choice(valid_directions) # choose once of the valid directions at random.

                # get a random sample of unused tiles
                unused_tile_pure = unused_tiles[unused_tiles != None]
                num_sample = np.min([20, len(unused_tile_pure)]) # by doing this random sampling, it ensures if the most compatable one it not the true fit, we have a chance of still getting the true fit
                samples = np.random.choice(unused_tile_pure,num_sample,replace=False)
                compat = self.cached_energies[element*np.ones_like(samples), samples, direction]
                max_compatability = samples[np.argmax(compat)]
                # add the chosen neighbor to the child

                if np.random.random() < self.mutation_probability: # random mutation
                    placement = np.random.choice(unused_tile_pure)
                else:
                    placement = max_compatability

                grid, num_used_tiles = self.place_tile(grid,direction,placement,used_tiles, unused_tiles, current_shape, num_used_tiles, m, n)

        return grid
    
    def place_tile(self, grid, direction, placement, used_tiles, unused_tiles, current_shape, num_used_tiles, m, n):
        if direction == 0: #"top"
            if m == 0: # the tile in child is already at the top
                grid = self.add_row_above(grid) # we don't need to worry about going over because we ensured that we chose directions open in that orientattion
                m += 1 # adding on the top, pushes the other indicies up
            grid[m-1,n] =  placement # pace the tile in approprate direction
            num_used_tiles += 1
            used_tiles.append(grid[m-1,n]) # add the used tile to the list
            unused_tiles[grid[m-1,n]] = None
        elif direction == 2: #'bottom' 
            if m == current_shape[0]-1: # the tile in child is already at the bottom
                grid = self.add_row_below(grid)
            grid[m+1,n] = placement # pace the tile in approprate direction
            num_used_tiles += 1
            used_tiles.append(grid[m+1,n]) # add the used tile to the list
            unused_tiles[grid[m+1,n]] = None
        elif direction == 1: #'left'
            if n == 0: # the tile in child is already at the top
                grid = self.add_column_left(grid)
                n += 1 # adding on the left pushes the other indicies up
            grid[m,n-1] = placement # pace the tile in approprate direction
            num_used_tiles += 1
            used_tiles.append(grid[m,n-1]) # add the used tile to the list
            unused_tiles[grid[m,n-1]] = None
        elif direction == 3:
            if n == current_shape[1]-1: # the tile in child is already at the top
                grid = self.add_column_right(grid)
            grid[m,n+1] = placement # pace the tile in approprate direction
            num_used_tiles += 1
            used_tiles.append(grid[m,n+1]) # add the used tile to the list
            unused_tiles[grid[m,n+1]] = None
        return grid, num_used_tiles
    
    def add_row_above(self, grid):
            num_cols = grid.shape[1]
            new_grid = np.vstack((-np.ones((1,num_cols),dtype=int), grid)) # whithout setting dtype=int it turns the entire array into a float but we need it as an int since these are indicies
            return new_grid

    def add_row_below(self,grid):
        num_cols = grid.shape[1]
        new_grid = np.vstack((grid,-np.ones((1,num_cols), dtype=int)))
        return new_grid

    def add_column_left(self,grid):
        num_rows = grid.shape[0]
        new_grid = np.hstack((-np.ones((num_rows,1),dtype=int), grid))
        return new_grid
    
    def add_column_right(self,grid):
        num_rows = grid.shape[0]
        new_grid = np.hstack((grid,-np.ones((num_rows,1), dtype=int)))
        return new_grid
    
    def check_open_side(self, grid, row:int, col:int, shape:tuple) -> tuple: # check this function seperately, it works perfectly (see ./python_tests/).
            '''special cases'''
            if shape == (1,1): # there is only a single element so all sides are open
                return {0:np.True_, 2:np.True_, 1:np.True_, 3:np.True_} # may as well use np.True_ to be consistant with the rest of the entries
            if shape[0] == 1: # the top and bottom must be open since there is only a single row
                '''left/right'''
                if (col == 0): # if on left boundary
                    left_open = not (shape[1] == self.grid_shape[1]) # if we are the maximal dimension, then top is not open
                    right_open = (grid[row,col+1] == -1)
                elif (col == shape[1]-1): # on right boundary
                    right_open = not (shape[1] == self.grid_shape[1])
                    left_open = (grid[row,col-1] == -1)
                else: # then we are in one of the middle rows
                    left_open = (grid[row,col-1] == -1)
                    right_open = (grid[row,col+1] == -1)
                return {0:np.True_, 2:np.True_, 1:left_open, 3:right_open}
            if shape[1] == 1: # left and right must be open
                '''top/bottom'''
                if (row == 0): # if on top boundary
                    top_open = not (shape[0] == self.grid_shape[0]) # if we are the maximal dimension, then top is not open
                    bottom_open = (grid[row+1,col] == -1)
                elif (row == shape[0]-1): # on bottom boundary
                    bottom_open = not (shape[0] == self.grid_shape[0])
                    top_open = (grid[row-1,col] == -1)
                else: # then we are in one of the middle rows
                    top_open = (grid[row-1,col] == -1)
                    bottom_open = (grid[row+1,col] == -1)
                return {0:top_open, 2:bottom_open, 1:np.True_, 3:np.True_}
            
            '''top/bottom'''
            if (row == 0): # if on top boundary
                top_open = not (shape[0] == self.grid_shape[0]) # if we the maximal dimension, then top is not open
                bottom_open = (grid[row+1,col] == -1)
            elif (row == shape[0]-1): # on bottom boundary
                bottom_open = not (shape[0] == self.grid_shape[0])
                top_open = (grid[row-1,col] == -1)
            else: # then we are in one of the middle rows
                top_open = (grid[row-1,col] == -1)
                bottom_open = (grid[row+1,col] == -1)
            
            '''left/right'''
            if (col == 0): # if on left boundary
                left_open = not (shape[1] == self.grid_shape[1]) # if we the maximal dimension, then top is not open
                right_open = (grid[row,col+1] == -1)
            elif (col == shape[1]-1): # on right boundary
                right_open = not (shape[1] == self.grid_shape[1])
                left_open = (grid[row,col-1] == -1)
            else: # then we are in one of the middle rows
                left_open = (grid[row,col-1] == -1)
                right_open = (grid[row,col+1] == -1)

            return {0:top_open, 2:bottom_open, 1:left_open, 3:right_open}
    
    def search_parents_for_same_adjacencies(self, parent1, parent2):
            '''
            In this function we search the parents for any equivalent adjacencies
            We return a list of dictionaries. Values of the dicts contain the adjacent member, if it is shared.
            We also need to store a list of all of the adjacencies so that we can choose one at random when filling the child. Repeats are fine - just think of it as weighting by the number of adjacencies that item has
            '''
            parent1 = parent1.ravel() # it will probably be easier to parse the flat arrays; probably fine to use ravel since I don't edit anything; .ravel is faster than .flat
            parent2 = parent2.ravel()
            cols = self.grid_shape[1]
            rows = self.grid_shape[0]

            list_of_tuples = [{0: None, 2: None, 1: None, 3: None} for _ in range(len(parent1))] # if it's in the 0 slot then it means that it is to the left of the entry the dictionary represents
            list_of_adjacencies = []

            for i in range(len(parent1)//2): # we can skip every other entry and still read each adjacency. (or we could read each one and only consider left and top adjacencies but that is probably less efficient since most of the time probably comes from the element in parent two)
                value = parent1[i]
                parent2_arg = np.argwhere(  parent2 == value)[0,0] # should find the point where parent2 takes the same value as parent 1 - probably done as efficintly as possible since there is a numpy function doing the work.
                if (i % cols != 0 and parent2_arg % cols != 0): # conditions for not having a neighbor on a particular side; both are not on top row
                    if parent1[i-1]==parent2[parent2_arg-1]:
                        neighbor = parent1[i-1]
                        list_of_tuples[parent1[i]][1] = neighbor
                        list_of_tuples[neighbor][3] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i >= cols and parent2_arg >= cols):
                    if parent1[i - cols] == parent2[parent2_arg - cols]:
                        neighbor = parent1[i-cols]
                        list_of_tuples[parent1[i]][0] = neighbor
                        list_of_tuples[neighbor][2] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i % cols != cols-1) and (parent2_arg % cols != cols-1):
                    if parent1[i+1] == parent2[parent2_arg+1]:
                        neighbor = parent1[i+1]
                        list_of_tuples[parent1[i]][3] = neighbor
                        list_of_tuples[neighbor][1] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i // cols != rows-1) and (parent2_arg // cols != rows-1):
                    if parent1[i+cols] == parent2[parent2_arg + cols]:
                        neighbor = parent1[i+cols]
                        list_of_tuples[parent1[i]][2] = neighbor
                        list_of_tuples[neighbor][0] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

            return list_of_tuples, list_of_adjacencies

#
#
#------------------------------------------
#
# I/O  Functions
#
#------------------------------------------
#

def generate_genome_from_file(filename="Inputs/Squirrel_Puzzle.jpg", grid_size=(8,8), color=True, energy_function = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y)), T0=10., Tf=0.5, geometric_decay_rate=0.9999, updates=False) -> simulation_grid:
    """
    Create a simulation_grid object directly from an image file.

    filename : string : path of the file to be imported
    grid_size : tuple of integers : gives the number of puzzle pieces which the image is split into
    color : boolean : Determines if the image is read in color or black and white
    """

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

    tiles = []

    if not color:
        width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
        length = gray_matrix.shape[0]
        tile_width = width//grid_size[1]
        tile_length = length//grid_size[0]
        #print(tile_length)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tiles.append({
                    0: gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)], 
                    2: gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)],
                    1: gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j],
                    3: gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1],
                    "entire": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)]# need this last one to reconstruct the array later
                })
        del gray_matrix # no need to store a large matrix any longer than we need it. We only need the boarders anyway
    else:
        width = color_volume.shape[1] # horizontal distance - should be the shoter of the two
        length = color_volume.shape[0]
        tile_width = width//grid_size[1]
        tile_length = length//grid_size[0]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tiles.append({
                    0: color_volume[tile_length*i,tile_width*j:tile_width*(j+1),:], # top
                    2: color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1),:], # bottom
                    1: color_volume[tile_length*i:tile_length*(i+1),tile_width*j,:], # left
                    3: color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1,:], # right
                    "entire": color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] # need this last one to reconstruct the array later
                }) # we use the wierd ordering so that we can use modular arithmatic to send top to bottom and left to right easily (and the reverse)

        del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

    num_tiles = grid_size[0]*grid_size[1] # total number of tiles in the image

    grid = np.arange(0,num_tiles,1,dtype=np.uint8).reshape((grid_size[0],grid_size[1])) # the representation of the image; using uint8 because nothing is negative or bigger than 255 and thus using any other integer system would be wasteful


    tiles = np.array(tiles, dtype=object) # apparently you can make a list of dictionaries into an array - this makes indexing later much easier - this is a change from the previous version

    cache_energies = np.zeros((num_tiles,num_tiles,4),dtype=float)

    for i in range(num_tiles):
        for j in range(num_tiles):
            for d_i in range(2):
                if i == j: # diagonal elements are set to infinite since they can never happen anyway
                    cache_energies[i,j,d_i] = np.inf
                else:
                    cache_energies[i,j,d_i] = energy_function( tiles[i][d_i], tiles[j][(d_i + 2) % 4] ) # although tiles could be indexed with an array, I think compatability would average over everything so We'll have to settle for the loop

    #Since we only did top and left, we can recover bottom and right since the matrix has the following property cache[i,j,0] = cache[j,i,2] and cache[i,j,1] = cache[j,i,3]
    # by only computing half of the directions in the loop we should halve the compute time of the loop

    X, Y = np.meshgrid(np.arange(0,num_tiles,1,dtype=np.uint8),np.arange(0,num_tiles,1,dtype=np.uint8))

    cache_energies[X,Y,2] = cache_energies[Y,X,0]

    cache_energies[X,Y,3] = cache_energies[Y,X,1]

    #I'm fairly happy with my idea to use meshgrid to do this.

    # this is actually suprisingly quick to compute though probably won't scale well for large puzzles. Luckily we only care about 64x64 right now.
    # in the current case it might actually take longer to open a read a file than just recompute all of the energies
    print("cached tile energies")
    

    return Genome([grid],tiles,cache_energies,numberGenerations=100,parentsPerGeneration=4,populationSize=1000,T0=T0, Tf=Tf, geometric_decay_rate=geometric_decay_rate,updates=updates)


def genome_reconstruct(simulation : Genome, color = True):
    tile_width = len(simulation.tile_data[0][0]) # length of the top of an arbitrary tile
    tile_length = len(simulation.tile_data[0][1]) # length of the left of an arbitrary tile
    
    width = tile_width * simulation.grid_shape[1] # horizontal distance - should be the shoter of the two
    length = tile_length * simulation.grid_shape[0]
    

    if color:
        resotred_page = np.zeros((length,width,3))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.product[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = simulation.tile_data[dict_index]["entire"]
    else:
        resotred_page = np.zeros((length,width))

        for i in range(simulation.grid_shape[0]):
            for j in range(simulation.grid_shape[1]):
                dict_index = simulation.product[i,j]
                resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = simulation.tile_data[dict_index]["entire"]

    return resotred_page.astype(np.uint8) # jpg can only handle this resolution anyway

def save_genome_output(filename, simulation : Genome, color = True, reconstruction = None):
    if reconstruction == None:
        resotred_page = reconstruct(simulation,color)

    cv2.imwrite(filename, resotred_page)
