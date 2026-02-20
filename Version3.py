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



