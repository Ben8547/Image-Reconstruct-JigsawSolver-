'''
Docstring for descrambler

Statement of problem:

We have now downloaded the manga images but there is a problem: they are scrambled, probably as a pirate protection.
We need to descramble the images.
The idea is as follows, each page seems to be but into very regular rectangles so we must first determin the dimensions of these rectnagles and determine where to cut our images.
Then load in an image and cut it according to the specified cuts.
We can then record the grayscale values on the top, bottom, left and right edges of each tile - the middle is irrelevant.
Finally match the pieces according to overlap of grayscale by comparing top and bottom, left and right.

WE CAN USE SIMMULATED ANNEALING!!!! (from PHY407) - determine a an energy function for the reassembled page and then minimize the energy function by swapping the tiles.
The energy function will somehow need to relate how well the sides of two tiles match. This seems much easier than having the program keep track of what pices fit together and fitting them together that way
other through some other brute force method - the only problem is that it may not be perfect - probably close enough though especially with a slow enough T schedule.
'''

from PIL import Image # for importing the image as an array
import numpy as np
import matplotlib.pyplot as plt
import cv2

if False: # test of loading the files into numpy
    # cv2
    gray_matrix = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    #PIL
    # Load image and convert to grayscale
    img = Image.open("test.jpg").convert("L")
    gray_matrix_PIL = np.array(img)

    fig, (ax_cv2,ax_PIL) = plt.subplots(1,2)
    ax_cv2.imshow(gray_matrix)
    ax_PIL.imshow(gray_matrix_PIL)
    plt.show()

    print("Image size is 1408x2112 pixel according to file properties")
    print(f"matrix dimensions in cv2: {gray_matrix.shape}")
    print(f"matrix dimensions in PIL: {gray_matrix_PIL.shape}")

# tests show that the each element of the array is a pixel so if we can figure out where the cut are pixelwise it should be easy to disect the image into pieces

'''Deicde color or non-color'''

color = True

'''Load in the test file (permanent)'''

file = "Original_RainbowFlower.jpg"

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

if False:  # test disection
    first_cut = gray_matrix[7*264:8*264,176*7:176*8]
    plt.imshow(first_cut)
    plt.show()

# since the tiles are casted 8x8 and both 1408 and 2112 are divisible by 8 it is the obvious choise to take the pixel locations of the cuts to be 176*n and 265*n respectvely
# this seems to have worked.

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
                "top": color_volume[tile_length*i:tile_length*(i+1),tile_width*j,:],
                "bottom": color_volume[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1,:],
                "left": color_volume[tile_length*i,tile_width*j:tile_width*(j+1),:],
                "right": color_volume[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1),:],
                "entire": color_volume[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] # need this last one to reconstruct the array later
            })

    del color_volume # no need to store a large matrix any longer than we need it. We only need the boarders anyway

'''
Now we need to construct an energy function for the system.
There are 8*8=64 tiles so we can model the system on an 8x8 grid. Then each tile occupies a single gridspace.
The interaction energy of a tile with its neighbors should cause the the most overlap in grayscale values between adjacent edges.
The first way I can think of to define the energy of interaction between two sides is the norm of the difference of their vectors - this seems naive but it is a good starting point.
'''

'''
I'd like to define some neat object to represen the grid points so that I can easily seatch for the neighbours but we might need to take a more 
brute force approach since I don't known much about class definitions and mechanics.

To create the grid, initial order of the dictionaries does not matter so we simply use put the indicies of the dictionaries in the list into an array and reshape that into a 8*8 grid.
'''

grid = np.arange(0,64,1).reshape((8,8))
#print(grid)
'''Then given an element of gird, we cn index the list tiles with that grid entry to get the corresponding tile data'''

from numpy.linalg import norm

if color:
    #compatability = lambda x,y: norm(x-y)
    compatability = lambda x,y: norm(x-y)
else:
    #compatability = lambda x,y: norm(x-y)
    compatability = lambda x,y: norm(x-y)

class simulation_grid: # the grid defined above is a member of this class when combined with it's list of dictionaries, these are utility functions to use on the simulation grid
    def __init__(self, grid, dict_list):
        self.gradient_cost = False # if true we use an energy function that penalizes changes in intensity gradients at edges, if false, we use the standard intensity cost
        self.simGrid = grid
        self.tile_data = dict_list
        self.grid_shape = self.simGrid.shape
        self.energy = self.total_energy() # set the total energy on creating the class

    def total_energy(self):
        energy = 0.
        for i in range(1,self.grid_shape[0]): # skip firt row
            for j in range(1,self.grid_shape[1]): # skip first column
                # we don't want to double count interactions so we first only compute the energies to the left and obove each point (skipping the topmost and leftmost row/column)
                # then since the edges do not interact we can stop here since each interacting edge has been counted exactly once.
                if not self.gradient_cost:
                    energy += self.interaction_energy((i,j))
                elif self.gradient_cost:
                    energy += self.gradient_interact_energy((i,j))
        return energy
    
    def gradient_interact_energy(self, grid_point:tuple) -> float:
        # finish this later: see Gallagher et al. for details
        return 

    def interaction_energy(self, grid_point:tuple) -> float:
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

        top_neighbor = self.tile_data[self.simGrid[row-1,column]]
        #bottom_neighbor = tile_data[sim_grid]
        left_neighbor = self.tile_data[self.simGrid[row,column-1]]
        #right_neighbor = self.tile_data[self.simGrid[row,column+1]]
        current = self.tile_data[self.simGrid[row,column]]

        # top side interaction
        top_energy = compatability(top_neighbor['bottom'], current['top'])
        # borrom side interaction
        #bottom_energy = norm(bottom_neighbor['top'] - current['bottom'])
        # top side interaction
        left_energy = compatability(left_neighbor['right'], current['left'])
        # top side interaction
        #right_energy = norm(right_neighbor['left'] - current['right'])

        return top_energy + left_energy# + bottom_energy + right_energy
    
    def markovStep(self, tempurature: float):
        '''
        Docstring for markovStep
        Mutates the self grid in accordance with the metropolis algorithm
        '''

        # find a point to swap with another
        point1 = (0,0)
        point2 = (0,0)
        count = 0
        '''
        I want to implement several differnent methods of purtutbation, not just swapping two random pieces, though this is certainly still a viable purtubation.
        I think we will have the following options:
            0. Swap two random tiles.
            1. choose row or column, then choose a random row or column and move all tiles from their initial side of the barrier to the other.
            2. choose a row and column, this divides the space into 4 regions. Then rotate the elements of a single region CCW in place.
            3. replace the current solution with a random permutation - this allows for a quick escape of a minimum if we are stuck; but requires getting fairly luck so might not be worth it
        Hopefully this provides enough extra ways of purturbing the system that we can search the solution space more efficiently.
        '''
        choice = np.random.randint(0,2)
        new_grid = np.copy(self.simGrid) # grid to store the purtubation in
        if choice == 0:
            while (point1 == point2) and (count < 10): # ensures (to an reasonable extent) that we swap two distinct points
                point1 = (np.random.randint(0,self.grid_shape[0]), np.random.randint(0,self.grid_shape[1]))
                point2 = (np.random.randint(0,self.grid_shape[0]), np.random.randint(0,self.grid_shape[1]))
                count += 1
            del count # no reason to keep it about 

            #replace the points in a new grid
            #temp_save = new_grid[point1] # there was no reason to ever do this - it's stored in the old grid still...
            new_grid[point1] = new_grid[point2]
            new_grid[point2] = self.simGrid[point1]
            #print(new_grid - self.simGrid) # debug

        elif choice == 1: # swap sides of grid
            if np.random.randint(0,2) == 0: # choose row
                row  = np.random.randint(1,self.grid_shape[0]) # choose random row index; don't allow first row since this will result in an unperturbed array
                new_grid[row:,:] = new_grid[:-row,:]
                new_grid[:row,:] = self.simGrid[-row:,:] # sets the first n rows equal the last n rows
                del row
            else: # choose column
                col = np.random.randint(1,self.grid_shape[1])
                new_grid[:,col:] = new_grid[:,:-col]
                new_grid[:,:col] = self.simGrid[:,-col:] # sets the first n rows equal the last n rows
                del col

        elif choice == 2: # rotate the pieces; currently off - seems like it would be difficult to recompute the energy in this case
            index = np.random.randint(1,self.grid_shape[1]) # needs to be square so we only use one index
            if np.random.randint(0,2) == 0:
                new_grid[:index,:index] = np.rot90(new_grid[:index,:index])
            else:
                new_grid[index:,index:] = np.rot90(new_grid[index:,index:])

            del index
        else: # currently off
            new_grid = np.random.permutation(new_grid)


        '''Now that we have purtubed the grid, we compute the energy and decide acceptance'''

        new = simulation_grid(new_grid,self.tile_data) # do this so that we can use the total energy function; probly not as quick as just writing a separate function, but saves me a lot of work

        previous_energy = self.energy
        new_energy = new.total_energy()
        del new # no reason to keep it around, the class stores a lot of redundant information
        #print(previous_energy) # debug
        #print(new_energy) # debug

        boltzmann_factor = np.exp((previous_energy - new_energy) / tempurature)

        if np.random.random() < boltzmann_factor: # always except whern previous >= new, sometimes accept an increase in the nergy - dig out of local minima.
            #print("accepted") # debug
            self.simGrid = new_grid
            self.energy = new_energy
    
    def reconstruct_page(self,schedule_constant:float, T0:float):
        # essentially just repeat the markov step while updating the tempurature.
        T = T0
        while T > 1:
            T *= schedule_constant
            print(f"energy: {self.energy}, Tempurature: {T}") # debug
            self.markovStep(T)
        return self.simGrid

page = simulation_grid(grid,tiles)

restored = page.reconstruct_page(0.9999,10.) # 0.9999, 200.

'''Now that we have the ordered array, all that remains is to put the grayscale map back together.'''

if color:
    resotred_page = np.zeros((length,width,3))

    for i in range(page.grid_shape[0]):
        for j in range(page.grid_shape[1]):
            dict_index = restored[i,j]
            resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = page.tile_data[dict_index]["entire"]

    resotred_page.astype(np.int8) # jpg can only handle this resolution anyway
    cv2.imwrite(f"annealing-color.jpg", resotred_page)
    resotred_page /= 255. # scale to the interval [0,1]; required for imshow
else:
    resotred_page = np.zeros((length,width))

    for i in range(page.grid_shape[0]):
        for j in range(page.grid_shape[1]):
            dict_index = restored[i,j]
            resotred_page[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = page.tile_data[dict_index]["entire"]
        cv2.imwrite(f"annealing-grayscale.jpg", resotred_page)

plt.imshow(resotred_page)
plt.show()