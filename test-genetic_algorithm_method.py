import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
Setup
'''

gray_matrix = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE) # load in the file

'''Chop up the image'''

width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
length = gray_matrix.shape[0]
tile_width = width//8
tile_length = length//8

tiles = []

for i in range(8):
    for j in range(8):
        tiles.append({
            "top": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j],
            "bottom": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1],
            "left": gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)],
            "right": gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)],
            "entire": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] # need this last one to reconstruct the array later
        })

del gray_matrix


'''
Genetic algorithm
'''

compatability = lambda x,y: -np.linalg.norm(x-y) # we are trying to maximize the fitness, so we return the negative since energy is minimized

class Geonome:
    def __init__(self, grid_list, dict_list):
        self.complementary_directions = { "top":"bottom", "bottom":"top", "left":"right", "right":"left"  }
        self.chromosomes = grid_list
        self.num_chromosomes = len(grid_list)
        self.num_tiles = len(dict_list)
        self.tile_data = dict_list
        self.grid_shape = grid_list[0].shape
        '''Now, when the class is defined, we compute all of the "best buddy" pieces. Since this is only dependent on the dict_list which does not mutate, we can do it once and leave it.'''
        #self.compatability_lists = self.generate_compatability_list()
        #self.best_buddies = self.find_best_buddies()
        #print('completed initialization of the geonome')

    '''def find_best_buddies(self):
        """
        A best buddy is defined as follows:

        Piece A is a best-buddy of piece B in direction R if for all X in PIECES, compatability(A,B;R) >= compatability(A,X;R)
        AND compatability(B,A;R^c) >= compatability(B,X;R^c)
        where R^c is the complementary direction of R.

        We store best buddies in the following way, the order of each piece is defined by self.tile_data. We define a new list of dictionaries where the keys are each
        of the 4 spatial directions. The value of eavh key is a list of the best buddies.
        Once the compatability lists are generated, this should theoretically be a rapd process since we already have the order of compababilities so once we find a 
        non-best buddy for an orientation we can move on.
        """
        list_dicts = []
        for k, dict in enumerate(self.compatability_lists): # since by definition of best beddies is a symmetric relation we can save some more computations, I probably won't take advantage of this for ease of coding. Later if I revisit then I can do that (in fact if it were reflexive then it would be an equivalence relation) 

            list_dicts.append({
                "top": [],
                "bottom": [],
                "left": [],
                "right": []
            })

        return list_dicts
    
    def generate_compatability_list(self):
        """
        This function compares each side of each tile against the complementary side of each other tile and compute the compatability score (using the same function as the fitness function)
        We store this information as list of dictionaries of lists. The dictionaries appear in the same order as they do in self.tile_data, the keys of each dictionary are the 4 spatial directions
        and the values of each key is a list of tuples of indicies of self.tile_data in order of compatability (least to greatest) and the associated compatabilty at that index.
        """
        list_dicts_lists = []
        for k, dict in enumerate(self.tile_data):
            top_compats = [compatability( dict['top'], self.tile_data[i][self.complementary_directions['top']] ) if i != k else -np.inf for i in range(self.num_tiles)]
            bottom_compats = [compatability( dict['bottom'], self.tile_data[i][self.complementary_directions['bottom']] ) if i != k else -np.inf for i in range(self.num_tiles)]
            left_compats = [compatability( dict['left'], self.tile_data[i][self.complementary_directions['left']] ) if i != k else -np.inf for i in range(self.num_tiles)]
            right_compats = [compatability( dict['right'], self.tile_data[i][self.complementary_directions['right']] ) if i != k else -np.inf for i in range(self.num_tiles)]

            list_dicts_lists.append({ # this could be made more efficient if we take into account that some of the below compatabilites are redundent, but that relies on the compatability funciton being a symmetric oporator which may not always be the case even though it is in our use case.
                "top":[ (j, top_compats[j]) for j in np.argsort( top_compats ) ],
                "bottom":[ (j, bottom_compats[j]) for j in (np.argsort( bottom_compats )) ],
                "left":[ (j, left_compats[j]) for j in (np.argsort( left_compats )) ],
                "right":[ (j, right_compats[j]) for j in (np.argsort( right_compats )) ]
            })
        return list_dicts_lists''' # removed because it was clunkey and probably not well optomized, I'd like to revisit when I more time to think about it

    def n_most_fit(self,n):
        if n > self.num_chromosomes:
            raise(ValueError)
        
        list_of_fitnesses = []
        for i in range(self.num_chromosomes):
            list_of_fitnesses.append(self.fitness(i))
        self.chromosomes = [self.chromosome[i] for i in np.argsort(list_of_fitnesses)] # order the chromosomes from least fit to most fit
        return self.chromosomes[-n-1:-1] # return the last n items of the list; will fail if n > length of chromosomes hence the earlier error message
    
    def fitness(self,i:int): #This method is analagous to the energy function in simmulated annealing
        chromosome = self.chromosomes[i]
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
    
    class Child:
        def __init__(self, parent1, parent2, outerself):
            self.outerself = outerself # "self" from the geonome class; allows us to access everything like the tile_data list and compatability_lists 
            self.max_shape = parent1.shape  # maximal dimensions of the child
            self.parents = (parent1,parent2)
            #create the child array. Since we need the initial placement to be anywhere, we will create a 1x1 matrix and then append rows and columns as we go, always preventing the array from being any larger than 8x8 (for the case of development)
            # the other easy alternative is to create an array that is 17 x 17 and then crop it at the end, but for larger puzzles that would require an unfeasable amount of memory, though would probably run faster.
            # Thus since it would probably scale better with the size of the puzzle we opt for the dynamic array.
            # we fill the array with -1s so that it does not confuse these empty spots with any of the indicies of self. tile_data
            self.grid = -np.ones((1,1),int)

            #seed the child with the first tile.
            # quickest way to search is to just go through every every other element of one parent, find the element in the other, and then check the neighbors of both in each direction to see if they match
            self.parent_adjacencies_lookup, self.parent_adjacencies = self.search_parents_for_same_adjacencies()
            # now if self.parent_adjacencies is non-empty we seed the array with one of it's members.
            if self.parent_adjacencies: # empty lists give false; lists with elements yield true
                self.grid[0,0] = np.random.choice(self.parent_adjacencies)
            else:
                self.grid[0,0] = np.random.choice(parent1.ravel())
            self.used_tiles = [None, self.grid[0,0]] # add the used tile to the list - this cannot be added again
            self.unused_tiles = [i for i in range(self.max_shape[0]*self.max_shape[1])]
            del self.unused_tiles[self.grid[0,0]]
            self.num_used_tiles = 1

            # now the child is seeded with its first element and can be filled uwing the aforementioned algorithm

            while self.num_used_tiles < len(parent1.ravel()):
                # start by searching for an element in the child in self.parent_adjacencies
                current_shape = self.grid.shape
                place_random = False
                if self.parent_adjacencies: # check that there even are duplicate adjacencies
                    dual_adjacent_in_child = []
                    for i, element in enumerate(self.grid.ravel()):
                        # i is the index of the raveled array; element is the integer representing the tile
                        m = i // current_shape[1] # row index of element
                        n = i % current_shape[1] # column index of element
                        if element in self.parent_adjacencies:
                            open = self.check_open_side(n,m, current_shape)
                            any_open = open['top'] or open['bottom'] or open['left'] or open['right']
                            if any_open: # check if the tile has an open adjacency
                                repetative_neighbors_dict = self.parent_adjacencies_lookup[element] # should return a dictionary with the parental repeated neighbors on each side listed
                                if (repetative_neighbors_dict['top'] not in self.used_tiles) or (repetative_neighbors_dict['bottom'] not in self.used_tiles) or (repetative_neighbors_dict['right'] not in self.used_tiles) or (repetative_neighbors_dict['left'] not in self.used_tiles): # check that the intended neighbor has not already been place in the child; recall that None is in the list of used tiles so we get True if there is no ajacency
                                    dual_adjacent_in_child.append(i) # this will be a list of all of the elements in the child that have repeated adjacencies in the parents that also have open sides in the grid
                    if dual_adjacent_in_child: # check the list is non-empty; i.e. were any of the neighbors not already chosen?
                        # choose a tile which a neighbour
                        tile = np.random.choice(dual_adjacent_in_child) # tile index in the flat array
                        m = tile // current_shape[0]
                        n = tile % current_shape[1]
                        tile = self.grid[m,n]
                        # now find its common neighbours in the parents, and choose one at random
                        repetative_neighbors_dict = self.parent_adjacencies_lookup[tile]
                        valid_directions = []
                        for direction in ['top','bottom','left','right']:
                            if (repetative_neighbors_dict[direction] != None) and self.check_open_side(n,m,current_shape)[direction]: # requires the element to be open in that direction and for it to actully have a valid neighbor
                                valid_directions.append(direction)
                        direction = np.random.choice(valid_directions) # choose a random valid direction
                        # place the tile in the chosen direction
                        if direction == 'top':
                            if n == 0: # the tile in child is already at the top
                                self.add_row_above() # we don't need to worry about going over because we ensured that we chose directions open in that orientattion
                            self.grid[n-1,m] = repetative_neighbors_dict[direction] # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[n-1,m]) # add the used tile to the list
                            del self.unused_tiles[self.grid[n-1,m]]
                        elif direction == 'bottom':
                            if n == current_shape[0]-1: # the tile in child is already at the bottom
                                self.add_row_below()
                            self.grid[n+1,m] = repetative_neighbors_dict[direction] # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[n+1,m]) # add the used tile to the list
                            del self.unused_tiles[self.grid[n+1,m]]
                        elif direction == 'left':
                            if m == 0: # the tile in child is already at the top
                                self.add_column_left()
                            self.grid[n,m-1] = repetative_neighbors_dict[direction] # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[n,m-1]) # add the used tile to the list
                            del self.unused_tiles[self.grid[n,m-1]]
                        elif direction == 'right':
                            if m == current_shape[1]-1: # the tile in child is already at the top
                                self.add_column_right()
                            self.grid[n,m+1] = repetative_neighbors_dict[direction] # pace the tile in approprate direction
                            self.num_used_tiles += 1
                            self.used_tiles.append(self.grid[n,m+1]) # add the used tile to the list
                            del self.unused_tiles[self.grid[n,m+1]]
                        
                    else: # there are no dual adjacencies present in the child
                        place_random = True

                else: #there are no dulpicate adjacencies 
                    place_random == True
                
                if place_random:
                    # find elements that have open side:
                    valid_elements = []
                    for i, element in enumerate(self.grid.ravel()):
                        if element != -1:
                            # i is the index of the raveled array; element is the integer representing the tile
                            m = i // current_shape[1] # row index of element
                            n = i % current_shape[1] # column index of element
                            open = self.check_open_side(n,m, current_shape)
                            any_open = open['top'] or open['bottom'] or open['left'] or open['right']
                            if any_open:
                                valid_elements.append(i)
                    element_index = np.random.choice(valid_elements) # choose one of the valid elements to add on to
                    m = i // current_shape[1]
                    n = i % current_shape[1]
                    element = self.grid[m,n]
                    # now find the valid neighbors of that element
                    for direction in ['top','bottom','left','right']:
                            if self.check_open_side(n,m,current_shape)[direction]: # requires the element to be open in that direction and for it to actully have a valid neighbor
                                valid_directions.append(direction)
                    direction = np.random.choice(valid_directions) # choose once of the valid directions at random.
                    
                    # get a random sample of unused tiles
                    num_sample = np.max([10,len(self.unused_tiles)]) # by doing this random sampling, it ensures if the most compatable one it not the true fit, we have a chance of still getting the true fit
                    samples = np.random.choice(self.unused_tiles,num_sample,replace=False)
                    max_compatability = (element,-np.inf)
                    for sample_tile in samples:
                        x = outerself.tile_data[element][direction]
                        y = outerself.tile_data[sample_tile][outerself.complementary_directions[direction]]
                        compat = compatability(x,y)
                        if compat > max_compatability[1]: # ensures at the end we have the most compatable one
                            max_compatability = (sample_tile,compat)
                    max_compatability = max_compatability[0]
                    # add the chosen neighbor to the child
                    if direction == 'top':
                        if n == 0: # the tile in child is already at the top
                            self.add_row_above() # we don't need to worry about going over because we ensured that we chose directions open in that orientattion
                        self.grid[n-1,m] = max_compatability # pace the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[n-1,m]) # add the used tile to the list
                        del self.unused_tiles[self.grid[n-1,m]]
                    elif direction == 'bottom':
                        if n == current_shape[0]-1: # the tile in child is already at the bottom
                            self.add_row_below()
                        self.grid[n+1,m] = max_compatability # pace the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[n+1,m]) # add the used tile to the list
                        del self.unused_tiles[self.grid[n+1,m]]
                    elif direction == 'left':
                        if m == 0: # the tile in child is already at the top
                            self.add_column_left()
                        self.grid[n,m-1] = max_compatability # pace the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[n,m-1]) # add the used tile to the list
                        del self.unused_tiles[self.grid[n,m-1]]
                    elif direction == 'right':
                        if m == current_shape[1]-1: # the tile in child is already at the top
                            self.add_column_right()
                        self.grid[n,m+1] = max_compatability # pace the tile in approprate direction
                        self.num_used_tiles += 1
                        self.used_tiles.append(self.grid[n,m+1]) # add the used tile to the list
                        del self.unused_tiles[self.grid[n,m+1]]
                    
                    ''' # choose a random element that is not -1
                    not_minus_1 = self.grid + 1
                    not_minus_1 = np.argwhere(not_minus_1) # returns the indicies where the child is not -1
                    tile = np.random.choice(not_minus_1)
                    element = self.grid[tile]''' # depreciated



        def check_open_side(self,row:int, col:int, shape) -> tuple:
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
            parent1 = self.parents[0].flat # it will probably be easier to parse the flat arrays
            parent2 = self.parents[1].flat
            cols = self.max_shape[1]
            rows = self.max_shape[0]

            list_of_tuples = [{"top": None, "bottom": None, "left": None, "right": None} for _ in range(len(parent1))] # if it's in the "top" slot then it means that it is to the left of the entry the dictionary represents
            list_of_adjacencies = []

            value = parent1[i]
            parent2_arg = np.argwhere(parent2 == value)[0,0] # should find the point where parent2 takes the same value as parent 1 - probably done as efficintly as possible since there is a numpy function doing the work.

            for i in range(len(parent1)//2): # we can skip every other entry and still read each adjacency. (or we could read each one and only consider left and top adjacencies but that is probably less efficient since most of the time probably comes from the element in parent two)
                if (i % cols != 0 and parent2_arg % cols != 0): # conditions for not having a neighbor on a particular side
                    if parent1[i-1]==parent2[parent2_arg-1]:
                        neighbor = parent1[i-1]
                        list_of_tuples[parent1[i]]["left"] = neighbor
                        list_of_tuples[neighbor]["right"] = parent1[i] # need to be able to look up both ways
                        list_of_adjacencies.append[parent1[i],neighbor]

                if (i >= cols and parent2_arg >= cols):
                    if parent1[i - cols] == parent2[parent2_arg - cols]:
                        neighbor = parent1[i-cols]
                        list_of_tuples[parent1[i]]["top"] = neighbor
                        list_of_tuples[neighbor]["bottom"] = parent1[i] # need to be able to look up both ways
                        list_of_adjacencies.append[parent1[i],neighbor]

                if (i % cols != cols-1) and (parent2_arg % cols != cols-1):
                    if parent1[i+1] == parent2[parent2_arg+1]:
                        neighbor = parent1[i+1]
                        list_of_tuples[parent1[i]]["right"] = neighbor
                        list_of_tuples[neighbor]["left"] = parent1[i] # need to be able to look up both ways
                        list_of_adjacencies.append[parent1[i],neighbor]

                if (i // cols != rows-1) and (parent2_arg // cols != rows-1):
                    if parent1[i+cols] == parent2[parent2_arg + cols]:
                        neighbor = parent1[i+cols]
                        list_of_tuples[parent1[i]]["bottom"] = neighbor
                        list_of_tuples[neighbor]["top"] = parent1[i] # need to be able to look up both ways
                        list_of_adjacencies.append[parent1[i],neighbor]

            return list_of_tuples, list_of_adjacencies
        
        def rand_parent(self):
            return self.parents[np.random.randint(0,2)]
        
        def add_row_above(self):
            num_cols = self.grid.shape[1]
            self.grid = np.vstack((-np.ones((1,num_cols)),self.grid))
            
        def add_row_below(self):
            num_cols = self.grid.shape[1]
            self.grid = np.vstack((self.grid,-np.ones((1,num_cols))))

        def add_column_left(self):
            num_rows = self.grid.shape[0]
            self.grid = np.hstack((-np.ones((num_rows,1)),self.grid))
        
        def add_column_right(self):
            num_rows = self.grid.shape[0]
            self.grid = np.hstack((self.grid,-np.ones((num_rows,1))))

            
        def mutate(self): # my original mutation function. I think this would be an interesting place to slot in simmulated annealing;
            # we can run a few iterations of a simmulated annealing algorithm to approve or reject the mutations.
            # the problem with our previous annealing program was that is could only move two tiles at a time, but here that behavior fits well
            # as we only want to make small purturbations with annealing and the large rearrangements are left to the crossing algorithm
            return
    
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

        # first step is to check if both parents share any common adjacencies. Easist way to do this is probably to iterate through one parent



        while child.num_used_tiles < self.num_tiles:
            self.chromosomes.append(child.mutate())
            self.num_chromosomes += 1

geonome = Geonome([np.arange(0,64,1).reshape((8,8))],tiles) # the collection of chromosomes; a list of arrays

'''
Generate Random Chromosomes (members of the solution space)
'''
num_chromosomes = 1000

for _ in range(num_chromosomes):
    geonome.chromosomes.append(np.random.permutation(geonome[0])) # takes in the array and randomly permutes the elements - this will generate our initial chromosomes.


'''
complete the solver
'''

num_generations = 100
num_initial_parents_per_gen = 4

for _ in range(num_chromosomes):
    geonome.chromosomes = geonome.n_most_fit(num_initial_parents_per_gen)
    geonome.num_chromosomes = num_initial_parents_per_gen
    while geonome.num_chromosomes < num_chromosomes:
        parent1 = np.random.randint(0,num_initial_parents_per_gen)
        parent2 = np.random.randint(0,num_initial_parents_per_gen)
        count = 0
        while parent1 == parent2 and count < 10: # prevent inbreeding
            count += 1 # prevents stalling here indefinately; if one repeat gets through it won't really hurt anything
            parent2 = np.random.randint(0,num_initial_parents_per_gen) # Since mutations will be introduced, inbreeding actually serves to preserve a parent with small modificatons; though given the way the current crossing algorithm works, I don't think that we should admit repeats
        geonome.cross(parent1,parent2) # appends a child to the list of chromosomes

'''
Now we have a selection 1000 crossbred products; we return the one with the best fitness
'''

best_child = geonome.n_most_fit(1)

# now we need to reassemble the image
resotred_page = np.zeros((length,width))

for i in range(geonome.grid_shape[0]):
    for j in range(geonome.grid_shape[1]):
        dict_index = best_child[i,j]
        resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = geonome.tile_data[dict_index]["entire"]

plt.imshow(resotred_page)
plt.show()
