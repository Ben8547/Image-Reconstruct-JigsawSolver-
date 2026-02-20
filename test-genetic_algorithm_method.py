import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep

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
                "top": color_volume[tile_length*i,tile_width*j:tile_width*(j+1),:],
                "bottom": color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1),:],
                "left": color_volume[tile_length*i:tile_length*(i+1),tile_width*j,:],
                "right": color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1,:],
                "entire": color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] # need this last one to reconstruct the array later
            })

    del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

'''
Genetic algorithm
'''

compatability = lambda x,y: np.mean((x-y)**2)#np.linalg.norm(x-y) # we are trying to maximize the fitness, so we return the negative since energy is minimized

class Genome:
    def __init__(self, grid_list, dict_list):
        self.complementary_directions = { "top":"bottom", "bottom":"top", "left":"right", "right":"left"  }
        self.T0 = 200. # for annealing
        self.temp_decay = 0.999 # for annealing
        self.directions = { "top", "bottom", "left", 'right' }
        self.chromosomes = grid_list
        self.num_chromosomes = len(grid_list)
        self.num_tiles = len(dict_list)
        self.tile_data = dict_list
        self.grid_shape = grid_list[0].shape
        self.mutation_probability = 1./1000. # i.e. one in 1000 additions will be random
        '''Now, when the class is defined, we compute all of the "best buddy" pieces. Since this is only dependent on the dict_list which does not mutate, we can do it once and leave it.'''
        #self.compatability_lists = self.generate_compatability_list()
        #self.best_buddies = self.find_best_buddies()
        #print('completed initialization of the genome')

    def n_most_fit(self,n:int):
        if n > self.num_chromosomes:
            raise(ValueError)
        
        list_of_fitnesses = []
        for i in range(self.num_chromosomes):
            list_of_fitnesses.append(self.fitness(i))
        self.chromosomes = [self.chromosomes[i] for i in np.argsort(list_of_fitnesses)] # order the chromosomes from least fit to most fit
        return self.chromosomes[-n:] # return the last n items of the list; will fail if n > length of chromosomes hence the earlier error message
    
    def fitness(self,i:int): #This method is analagous to the energy function in simmulated annealing
        chromosome = self.chromosomes[i]
        energy = 0.
        for i in range(1,self.grid_shape[0]): # skip firt row
            for j in range(1,self.grid_shape[1]): # skip first column
                # we don't want to double count interactions so we first only compute the energies to the left and obove each point (skipping the topmost and leftmost row/column)
                # then since the edges do not interact we can stop here since each interacting edge has been counted exactly once.
                energy += self.interaction_energy(chromosome,(i,j))
        return -energy
    
    def energy(self, grid): # just negative fitness, but allows passing a grid and not just an index so can effect non-chromosome arrays
        chromosome = grid
        energy = 0.
        for i in range(1,self.grid_shape[0]): # skip firt row
            for j in range(1,self.grid_shape[1]): # skip first column
                # we don't want to double count interactions so we first only compute the energies to the left and obove each point (skipping the topmost and leftmost row/column)
                # then since the edges do not interact we can stop here since each interacting edge has been counted exactly once.
                energy += self.interaction_energy(chromosome,(i,j))
        return energy
    
    def interaction_energy(self, simGrid, grid_point:tuple) -> float:
        '''simGrid is an artifact from this begin from the simmulated annealing trials - it is an indevidual chromosome in this class'''
        row = grid_point[0]
        column = grid_point[1]

        # for why we only count the top and right neighbors, see the above function "total_energy" - in short it precents double counting

        top_neighbor = self.tile_data[simGrid[row-1,column]]
        left_neighbor = self.tile_data[simGrid[row,column-1]]
        current = self.tile_data[simGrid[row,column]]

        # top side interaction
        top_energy = compatability(top_neighbor['bottom'], current['top'])
        # left side interaction
        left_energy = compatability(left_neighbor['right'], current['left'])

        return top_energy + left_energy
    
    def markovStep(self, chromosome, tempurature: float):
        '''Apply a single step of the annealing algorithm to the chromosome'''
         # find a point to swap with another
        point1 = (0,0)
        point2 = (0,0)
        count = 0

        choice = np.random.randint(0,3)
        new_grid = np.copy(chromosome) # grid to store the purtubation in
        if choice == 0:
            while (point1 == point2) and (count < 10): # ensures (to an reasonable extent) that we swap two distinct points
                point1 = (np.random.randint(0,self.grid_shape[0]), np.random.randint(0,self.grid_shape[1]))
                point2 = (np.random.randint(0,self.grid_shape[0]), np.random.randint(0,self.grid_shape[1]))
                count += 1
            del count # no reason to keep it about 

            #replace the points in a new grid
            #temp_save = new_grid[point1] # there was no reason to ever do this - it's stored in the old grid still...
            new_grid[point1] = new_grid[point2]
            new_grid[point2] = chromosome[point1]
            #print(new_grid - self.simGrid) # debug

        elif choice == 1: # swap sides of grid
            if np.random.randint(0,2) == 0: # choose row
                row  = np.random.randint(1,self.grid_shape[0]) # choose random row index; don't allow first row since this will result in an unperturbed array
                new_grid[row:,:] = new_grid[:-row,:]
                new_grid[:row,:] = chromosome[-row:,:] # sets the first n rows equal the last n rows
                del row
            else: # choose column
                col = np.random.randint(1,self.grid_shape[1])
                new_grid[:,col:] = new_grid[:,:-col]
                new_grid[:,:col] = chromosome[:,-col:] # sets the first n rows equal the last n rows
                del col

        elif choice == 2: # rotate the pieces
            index = np.random.randint(1,self.grid_shape[1]) # needs to be square so we only use one index
            if np.random.randint(0,2) == 0:
                new_grid[:index,:index] = np.rot90(new_grid[:index,:index])
            else:
                new_grid[index:,index:] = np.rot90(new_grid[index:,index:])

            del index
        
        ''' compute the energy (negative fitness) '''

        previous_energy = self.energy(chromosome)
        new_energy = self.energy(new_grid)

        boltzmann_factor = np.exp((previous_energy - new_energy) / tempurature)

        if np.random.random() < boltzmann_factor: # always except whern previous >= new, sometimes accept an increase in the nergy - dig out of local minima.
            #print("accepted") # debug
            return new_grid, new_energy
        else: return chromosome, previous_energy
    
    def anneal(self, chromosome, T0, decay_rate):
        '''apply simulated annealing to a single chromosome'''
        T = T0
        while T > 1.:
            chromosome, energy = self.markovStep(chromosome,T)
            T *= decay_rate
            print(f"Energy: {energy}, Temperature: {T}") # to view progress, not vital
        return chromosome # should mutate in place, but I'll return for asureadness
    
    def anneal_all(self):
        ''' Take all of the chromosomes and run them through simmulated annealing, then replace the chromosomes '''
        for i, chromosome in enumerate(self.chromosomes):
            self.chromosomes[i] = self.anneal(chromosome,self.T0, self.temp_decay) # replace the chromosome

    def estimate_T0(self,target_acceptance_rate=0.1, num_samples = 10):
        '''Add explaination later'''
        # estimate a T0 for the given state of the chromosomes
        chromosome = self.chromosomes[-1] # choose the most fit chromosome
        deltas = []
        base_energy = self.energy(chromosome)
        for _ in range(num_samples):
            trial, new_energy = self.markovStep(chromosome, tempurature=1e9)  # force acceptance with high tempurature
            deltas.append(abs(new_energy - base_energy))

        mean_delta = np.mean(deltas)
        Temperature = -mean_delta / np.log(target_acceptance_rate)
        
        return Temperature
    
    class Child:
        def __init__(self, parent1, parent2, outerself):
            self.outerself = outerself # "self" from the genome class; allows us to access everything like the tile_data list and compatability_lists 
            self.mutation_probability = outerself.mutation_probability
            self.mutation_T0 = 50.
            self.mutation_decay = 0.95
            self.max_shape = parent1.shape  # maximal dimensions of the child
            self.parents = (parent1,parent2)
            #create the child array. Since we need the initial placement to be anywhere, we will create a 1x1 matrix and then append rows and columns as we go, always preventing the array from being any larger than 8x8 (for the case of development)
            # the other easy alternative is to create an array that is 17 x 17 and then crop it at the end, but for larger puzzles that would require an unfeasable amount of memory, though would probably run faster.
            # Thus since it would probably scale better with the size of the puzzle we opt for the dynamic array.
            # we fill the array with -1s so that it does not confuse these empty spots with any of the indicies of self. tile_data
            self.grid = -np.ones((1,1),dtype=int)

            #seed the child with the first tile.
            # quickest way to search is to just go through every every other element of one parent, find the element in the other, and then check the neighbors of both in each direction to see if they match
            self.parent_adjacencies_lookup, self.parent_adjacencies = self.search_parents_for_same_adjacencies()
            # now if self.parent_adjacencies is non-empty we seed the array with one of it's members.
            if self.parent_adjacencies: # empty lists give false; lists with elements yield true
                self.grid[0,0] = np.random.choice(self.parent_adjacencies)
            else: # no adjacencies
                self.grid[0,0] = np.random.choice(parent1.ravel())
            self.used_tiles = [None, self.grid[0,0]] # add the used tile to the list - this cannot be added again
            self.unused_tiles = np.array([i for i in range(self.max_shape[0]*self.max_shape[1])],dtype=object) # this one specifically is an array because we need to use a mask on it later
            self.unused_tiles[self.grid[0,0]] = None
            self.num_used_tiles = 1

            # now the child is seeded with its first element and can be filled uwing the aforementioned algorithm

            #print(self.parent_adjacencies) #debug
            while self.num_used_tiles < len(parent1.ravel()):
                #print(self.grid) # debug
                # start by searching for an element in the child in self.parent_adjacencies
                current_shape = self.grid.shape
                place_random = False
                if self.parent_adjacencies: # check that there even are duplicate adjacencies
                    #print(self.parent_adjacencies) # debug
                    dual_adjacent_in_child = []
                    for i, element in enumerate(self.grid.ravel()):
                        # i is the index of the raveled array; element is the integer representing the tile
                        m = i // current_shape[1] # row index of element
                        n = i % current_shape[1] # column index of element
                        if element in self.parent_adjacencies:
                            open = self.check_open_side(m,n, current_shape)
                            repetative_neighbors_dict = self.parent_adjacencies_lookup[element] # should return a dictionary with the parental repeated neighbors on each side listed
                            if any( ( (repetative_neighbors_dict[d] not in self.used_tiles) and ( open[d] != False ) ) for d in outerself.directions ): # check that the intended neighbor has not already been place in the child; recall that None is in the list of used tiles so we get True if there is no ajacency
                                dual_adjacent_in_child.append(i) # this will be a list of all of the elements in the child that have repeated adjacencies in the parents that also have open sides in the grid
                            else: # then the partner of this tile is used already - there is no reason to consider either or them in future loops
                                self.parent_adjacencies.remove(element) # no reason to consider this element again on future loops
                    if dual_adjacent_in_child: # check the list is non-empty; i.e. were any of the neighbors not already chosen?
                        # choose a tile which a neighbour
                        tile = np.random.choice(dual_adjacent_in_child) # tile index in the flat array
                        m = tile // current_shape[1] # row of tile
                        n = tile % current_shape[1] # column of tile
                        tile = self.grid[m,n]
                        # now find its common neighbours in the parents, and choose one at random
                        repetative_neighbors_dict = self.parent_adjacencies_lookup[tile]
                        valid_directions = []
                        for direction in ['top','bottom','left','right']:
                            if (repetative_neighbors_dict[direction] != None) and self.check_open_side(m,n,current_shape)[direction]: # requires the element to be open in that direction and for it to actully have a valid neighbor
                                valid_directions.append(direction)
                        direction = np.random.choice(valid_directions) # choose a random valid direction

                        if np.random.random() < self.mutation_probability: # random mutation
                            placement = np.random.choice(self.unused_tiles[self.unused_tiles != None])
                        else:
                            placement = repetative_neighbors_dict[direction]

                        # place the tile in the chosen direction
                        if direction == 'top':
                            if m == 0: # the tile in child is already at the top
                                self.add_row_above() # we don't need to worry about going over because we ensured that we chose directions open in that orientattion
                                m += 1
                            self.grid[m-1,n] =  placement # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[m-1,n]) # add the used tile to the list
                            self.unused_tiles[self.grid[m-1,n]] = None
                        elif direction == 'bottom':
                            if m == current_shape[0]-1: # the tile in child is already at the bottom
                                self.add_row_below()
                            self.grid[m+1,n] = placement # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[m+1,n]) # add the used tile to the list
                            self.unused_tiles[self.grid[m+1,n]] = None
                        elif direction == 'left':
                            if n == 0: # the tile in child is already at the top
                                self.add_column_left()
                                n += 1
                            self.grid[m,n-1] = placement # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[m,n-1]) # add the used tile to the list
                            self.unused_tiles[self.grid[m,n-1]] = None
                        elif direction == 'right':
                            if n == current_shape[1]-1: # the tile in child is already at the top
                                self.add_column_right()
                            self.grid[m,n+1] = placement # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[m,n+1]) # add the used tile to the list
                            self.unused_tiles[self.grid[m,n+1]] = None
                        
                    else: # there are no dual adjacencies present in the child
                        place_random = True

                else: #there are no dulpicate adjacencies 
                    place_random = True
                
                if place_random:
                    # find elements that have open side:
                    valid_elements = []
                    for i, element in enumerate(self.grid.ravel()):
                        if element != -1:
                            # i is the index of the raveled array; element is the integer representing the tile
                            m = i // current_shape[1] # row index of element
                            n = i % current_shape[1] # column index of element
                            open = self.check_open_side(m,n, current_shape)
                            any_open = open['top'] or open['bottom'] or open['left'] or open['right']
                            if any_open:
                                valid_elements.append(i)
                    element_index = np.random.choice(valid_elements) # choose one of the valid elements to add on to
                    m = element_index // current_shape[1]
                    n = element_index % current_shape[1]
                    element = self.grid[m,n]
                    # now find the valid neighbors of that element
                    valid_directions = []
                    for direction in ['top','bottom','left','right']:
                            if self.check_open_side(m,n,current_shape)[direction]: # requires the element to be open in that direction and for it to actully have a valid neighbor
                                valid_directions.append(direction)
                    direction = np.random.choice(valid_directions) # choose once of the valid directions at random.
                    
                    # get a random sample of unused tiles
                    unused_tile_pure = self.unused_tiles[self.unused_tiles != None]
                    #print(unused_tile_pure) # debug
                    num_sample = np.min([20, len(unused_tile_pure)]) # by doing this random sampling, it ensures if the most compatable one it not the true fit, we have a chance of still getting the true fit
                    samples = np.random.choice(unused_tile_pure,num_sample,replace=False)
                    max_compatability = (element,-np.inf)
                    for sample_tile in samples:
                        x = outerself.tile_data[element][direction]
                        y = outerself.tile_data[sample_tile][outerself.complementary_directions[direction]]
                        compat = compatability(x,y)
                        if compat > max_compatability[1]: # ensures at the end we have the most compatable one
                            max_compatability = (sample_tile,compat)
                    max_compatability = max_compatability[0]
                    # add the chosen neighbor to the child

                    if np.random.random() < self.mutation_probability: # random mutation
                        placement = np.random.choice(unused_tile_pure)
                    else:
                        placement = max_compatability

                    if direction == 'top':
                        if m == 0: # the tile in child is already at the top
                            self.add_row_above() # we don't need to worry about going over because we ensured that we chose directions open in that orientattion
                            m += 1
                        self.grid[m-1,n] = placement # place the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[m-1,n]) # add the used tile to the list
                        self.unused_tiles[self.grid[m-1,n]] = None
                    elif direction == 'bottom':
                        if m == current_shape[0]-1: # the tile in child is already at the bottom
                            self.add_row_below()
                        self.grid[m+1,n] = placement # place the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[m+1,n]) # add the used tile to the list
                        self.unused_tiles[self.grid[m+1,n]] = None
                    elif direction == 'left':
                        if n == 0: # the tile in child is already at the top
                            n += 1
                            self.add_column_left()
                        self.grid[m,n-1] = placement # place the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[m,n-1]) # add the used tile to the list
                        self.unused_tiles[self.grid[m,n-1]] = None
                    elif direction == 'right':
                        if n == current_shape[1]-1: # the tile in child is already at the top
                            self.add_column_right()
                        self.grid[m,n+1] = placement # place the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[m,n+1]) # add the used tile to the list
                        self.unused_tiles[self.grid[m,n+1]] = None
                    
                    ''' # choose a random element that is not -1
                    not_minus_1 = self.grid + 1
                    not_minus_1 = np.argwhere(not_minus_1) # returns the indicies where the child is not -1
                    tile = np.random.choice(not_minus_1)
                    element = self.grid[tile]''' # depreciate

            # end of while loop
            self.mutate() # once all of the tiles have been placed, run the mutation


        def check_open_side(self,row:int, col:int, shape:tuple) -> tuple: # check this function seperately, it works perfectly (see ./python_tests/).
            '''special cases'''
            if shape == (1,1): # there is only a single element so all sides are open
                return {'top':np.True_, 'bottom':np.True_, 'left':np.True_, 'right':np.True_} # may as well use np.True_ to be consistant with the rest of the entries
            if shape[0] == 1: # the top and bottom must be open since there is only a single row
                '''left/right'''
                if (col == 0): # if on left boundary
                    left_open = not (shape[1] == self.max_shape[1]) # if we are the maximal dimension, then top is not open
                    right_open = (self.grid[row,col+1] == -1)
                elif (col == shape[1]-1): # on right boundary
                    right_open = not (shape[1] == self.max_shape[1])
                    left_open = (self.grid[row,col-1] == -1)
                else: # then we are in one of the middle rows
                    left_open = (self.grid[row,col-1] == -1)
                    right_open = (self.grid[row,col+1] == -1)
                return {'top':np.True_, 'bottom':np.True_, 'left':left_open, 'right':right_open}
            if shape[1] == 1: # left and right must be open
                '''top/bottom'''
                if (row == 0): # if on top boundary
                    top_open = not (shape[0] == self.max_shape[0]) # if we are the maximal dimension, then top is not open
                    bottom_open = (self.grid[row+1,col] == -1)
                elif (row == shape[0]-1): # on bottom boundary
                    bottom_open = not (shape[0] == self.max_shape[0])
                    top_open = (self.grid[row-1,col] == -1)
                else: # then we are in one of the middle rows
                    top_open = (self.grid[row-1,col] == -1)
                    bottom_open = (self.grid[row+1,col] == -1)
                return {'top':top_open, 'bottom':bottom_open, 'left':np.True_, 'right':np.True_}
            
            '''top/bottom'''
            if (row == 0): # if on top boundary
                top_open = not (shape[0] == self.max_shape[0]) # if we the maximal dimension, then top is not open
                bottom_open = (self.grid[row+1,col] == -1)
            elif (row == shape[0]-1): # on bottom boundary
                bottom_open = not (shape[0] == self.max_shape[0])
                top_open = (self.grid[row-1,col] == -1)
            else: # then we are in one of the middle rows
                top_open = (self.grid[row-1,col] == -1)
                bottom_open = (self.grid[row+1,col] == -1)
            
            '''left/right'''
            if (col == 0): # if on left boundary
                left_open = not (shape[1] == self.max_shape[1]) # if we the maximal dimension, then top is not open
                right_open = (self.grid[row,col+1] == -1)
            elif (col == shape[1]-1): # on right boundary
                right_open = not (shape[1] == self.max_shape[1])
                left_open = (self.grid[row,col-1] == -1)
            else: # then we are in one of the middle rows
                left_open = (self.grid[row,col-1] == -1)
                right_open = (self.grid[row,col+1] == -1)

            return {'top':top_open, 'bottom':bottom_open, 'left':left_open, 'right':right_open}

        def search_parents_for_same_adjacencies(self):
            '''
            In this function we search the parents for any equivalent adjacencies
            We return a list of dictionaries. Values of the dicts contain the adjacent member, if it is shared.
            We also need to store a list of all of the adjacencies so that we can choose one at random when filling the child. Repeats are fine - just think of it as weighting by the number of adjacencies that item has
            '''
            parent1 = self.parents[0].ravel() # it will probably be easier to parse the flat arrays; probably fine to use ravel since I don't edit anything; .ravel is faster than .flat
            parent2 = self.parents[1].ravel()
            cols = self.max_shape[1]
            rows = self.max_shape[0]

            list_of_tuples = [{"top": None, "bottom": None, "left": None, "right": None} for _ in range(len(parent1))] # if it's in the "top" slot then it means that it is to the left of the entry the dictionary represents
            list_of_adjacencies = []

            for i in range(len(parent1)//2): # we can skip every other entry and still read each adjacency. (or we could read each one and only consider left and top adjacencies but that is probably less efficient since most of the time probably comes from the element in parent two)
                value = parent1[i]
                parent2_arg = np.argwhere(  parent2 == value)[0,0] # should find the point where parent2 takes the same value as parent 1 - probably done as efficintly as possible since there is a numpy function doing the work.
                if (i % cols != 0 and parent2_arg % cols != 0): # conditions for not having a neighbor on a particular side
                    if parent1[i-1]==parent2[parent2_arg-1]:
                        neighbor = parent1[i-1]
                        list_of_tuples[parent1[i]]["left"] = neighbor
                        list_of_tuples[neighbor]["right"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i >= cols and parent2_arg >= cols):
                    if parent1[i - cols] == parent2[parent2_arg - cols]:
                        neighbor = parent1[i-cols]
                        list_of_tuples[parent1[i]]["top"] = neighbor
                        list_of_tuples[neighbor]["bottom"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i % cols != cols-1) and (parent2_arg % cols != cols-1):
                    if parent1[i+1] == parent2[parent2_arg+1]:
                        neighbor = parent1[i+1]
                        list_of_tuples[parent1[i]]["right"] = neighbor
                        list_of_tuples[neighbor]["left"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

                if (i // cols != rows-1) and (parent2_arg // cols != rows-1):
                    if parent1[i+cols] == parent2[parent2_arg + cols]:
                        neighbor = parent1[i+cols]
                        list_of_tuples[parent1[i]]["bottom"] = neighbor
                        list_of_tuples[neighbor]["top"] = parent1[i] # need to be able to look up both ways
                        if parent1[i] not in list_of_adjacencies:
                            list_of_adjacencies.append(parent1[i])
                        if neighbor not in list_of_adjacencies:
                            list_of_adjacencies.append(neighbor)

            #print(list_of_adjacencies) # debug
            #print(self.parents[0])
            #print(self.parents[1])

            return list_of_tuples, list_of_adjacencies
        
        def rand_parent(self):
            return self.parents[np.random.randint(0,2)]
        
        def add_row_above(self):
            num_cols = self.grid.shape[1]
            self.grid = np.vstack((-np.ones((1,num_cols),dtype=int), self.grid)) # whithout setting dtype=int it turns the entire array into a float but we need it as an int since these are indicies
            
        def add_row_below(self):
            num_cols = self.grid.shape[1]
            self.grid = np.vstack((self.grid,-np.ones((1,num_cols), dtype=int)))

        def add_column_left(self):
            num_rows = self.grid.shape[0]
            self.grid = np.hstack((-np.ones((num_rows,1),dtype=int), self.grid))
        
        def add_column_right(self):
            num_rows = self.grid.shape[0]
            self.grid = np.hstack((self.grid,-np.ones((num_rows,1), dtype=int)))

            
        def mutate(self): # my original mutation function. I think this would be an interesting place to slot in simmulated annealing;
            # we can run a few iterations of a simmulated annealing algorithm to approve or reject the mutations.
            # the problem with our previous annealing program was that is could only move two tiles at a time, but here that behavior fits well
            # as we only want to make small purturbations with annealing and the large rearrangements are left to the crossing algorithm
            if False:
                self.grid = self.outerself.anneal(self.grid,self.mutation_T0, self.mutation_decay) # seems to add a lot of time for not a whole lot of lot of efficacy
            pass

    
    def cross(self, parent1:int, parent2:int): # this function crosses two parent chromoseoms to create a child
        '''
        parent1 and parent 2 are indicies of the list self.chromosomes
        A good crossover function should have a few heuristic properties (I think; I didn't actually read this anywhere):
            1. Parent1 X Parent1 = Parent1 + mutation; i.e. before mutations, the oporations should be idempotent before mutation
        The mutator from Sholomon et al. (see the papers folder in this directory) takes the following steps that we also adhere to (if a piece is in the child it cannot be placed again):
            0. look at both parents, if they share any adjacencies (i.e. if tile1 is to the left of tile4 in both parents) then choose a tile in that adjacency at random and place it. If not, choose a random tile and place it.
            repeat the following until all tiles are places
            1. look if both parents share any adjacencies with any of the placed tile. If so, choose a random adjacency and place it into the child.
               Do this for only one pairing (randomly chosen out of all possibibilities).
               If no such pieces exist, skip to phase 2.
            2. Choose a parent and check if there is a piece in the parent currently in the child with an adjacent side that is adjacent to it's best buddy in the parent. If there is then place that piece
               into the corre If one exists in the open spatial configuration. If not then choose a random piece with adjacencies in the child
               and put a pice with the highest compatability to an open side adjacent to that side; I am not going to do the best buddies search right now (instead just check that the adjecent piece is highest compatability in one direction), I feel like using raw compatability scores shoudd suffice, I might come back on another day and add this in.
            3. Mutation - instead of placing the determined piece, place a random piece with low probability - I may decide to incorperate this later, but I will certainly use my annealing mutator

        This is what I want to do instead:
            0. The same as above (seeding)
            1.  again, look for adjacencies with placed tiles. This time, if none exist then choose a random sample of tiles, compute the scores, choose the best match, and place.
            3. run a few iterations of simulated annealing to mutate the array (low initial tempurature)
        '''
        parent1 = self.chromosomes[parent1]
        parent2 = self.chromosomes[parent2]

        # create the offspring according to the algorithm
        child = self.Child(parent1,parent2, self)

        self.chromosomes.append(child.grid) # add the child to the chromosomes
        self.num_chromosomes += 1

genome = Genome([np.arange(0,64,1,dtype=int).reshape((8,8))],tiles) # the collection of chromosomes; a list of arrays; jpg can only support int 8 so no need to use anything fancier

'''
Generate Random Chromosomes (members of the solution space)
'''
num_chromosomes = 500 # should set to 1000

'''for _ in range(num_chromosomes):
    genome.chromosomes.append(np.random.permutation(genome.chromosomes[0].ravel()).reshape(genome.chromosomes[0].shape)) # takes in the array and randomly permutes the elements - this will generate our initial chromosomes.
genome.num_chromosomes = num_chromosomes''' # method that the paper uses; I think I can do better by seeding with simulated annealing

num_chromosomes_to_seed_with = 10 #10

for _ in range(num_chromosomes_to_seed_with):
    genome.chromosomes.append(np.random.permutation(genome.chromosomes[0].ravel()).reshape(genome.chromosomes[0].shape)) # takes in the array and randomly permutes the elements - this will generate our initial chromosomes.
genome.num_chromosomes = num_chromosomes_to_seed_with

genome.anneal_all() # replace all of the chromosome with the annealed version
print("initial annealing complete")

'''
complete the solver
'''

num_generations = 10 # should set to 100
num_initial_parents_per_gen = 4 # should set to 4

generation = 1

genome.T0 = genome.estimate_T0(0.1,10) # set T0 to much lower now that we are done with the initial seeding; don't want to reverse our progress
print(f"T0 set to {genome.T0}")
sleep(4) # give time to read output

show = False # True is I plan to watch the simulation - set to false otherwise

while generation <= num_generations:
    genome.chromosomes = genome.n_most_fit(num_initial_parents_per_gen) # get the initial parents
    genome.T0 = genome.estimate_T0(0.1,10) # set T0 to much lower now that we are done with the initial seeding; don't want to reverse our progress
    print(f"T0 set to {genome.T0}")
    genome.anneal_all() # anneal the parents
    genome.num_chromosomes = len(genome.chromosomes)
    if show:
        genome.chromosomes = genome.n_most_fit(num_initial_parents_per_gen) # resort after annealing
        print(f"Best starting energy of generation {generation} is {-genome.fitness(-1)}")
        print(f"Worst starting energy of generation {generation} is {-genome.fitness(0)}")
        sleep(2) # give time to read - should comment out if not watching
    while genome.num_chromosomes < num_chromosomes:
        print(f"generation: {generation}, chromosome count: {genome.num_chromosomes}") # to see progress, mainly for debugging
        parent1 = np.random.randint(0,num_initial_parents_per_gen)
        parent2 = np.random.randint(0,num_initial_parents_per_gen)
        count = 0
        while parent1 == parent2 and count < 10: # prevent inbreeding
            count += 1 # prevents stalling here indefinately; if one repeat gets through it won't really hurt anything
            parent2 = np.random.randint(0,num_initial_parents_per_gen) # Since mutations will be introduced, inbreeding actually serves to preserve a parent with small modificatons; though given the way the current crossing algorithm works, I don't think that we should admit repeats
        genome.cross(parent1,parent2) # appends a child to the list of chromosomes
    generation += 1

'''
Now we have a selection 1000 crossbred products; we return the one with the best fitness
'''

best_child = genome.n_most_fit(1)[0] # this also order the chromosomes from least to most fit so we can get the fitness by just appling class fitness function to the last chromosome
final_energy = -genome.fitness(-1)

print(f"final energy: {final_energy}")


# now we need to reassemble the image
if color:
    resotred_page = np.zeros((length,width,3))

    for i in range(genome.grid_shape[0]):
        for j in range(genome.grid_shape[1]):
            dict_index = best_child[i,j]
            resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = genome.tile_data[dict_index]["entire"]

    resotred_page = resotred_page.astype(np.uint8) # jpg can only handle this resolution anyway
    cv2.imwrite(f"genetic-color-{generation-1}-Generations.jpg", resotred_page)
else:
    resotred_page = np.zeros((length,width))

    for i in range(genome.grid_shape[0]):
        for j in range(genome.grid_shape[1]):
            dict_index = best_child[i,j]
            resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = genome.tile_data[dict_index]["entire"]
        resotred_page = resotred_page.astype(np.uint8)
        cv2.imwrite(f"genetic-grayscale-{generation-1}-Generations.jpg", resotred_page)
        

plt.imshow(resotred_page)
plt.show()
