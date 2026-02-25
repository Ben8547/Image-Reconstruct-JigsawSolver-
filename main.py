from Version3 import generate_simGrid_from_file
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import cv2

'''
Setup
'''

file = "Inputs/"+"test.jpg"

''' Energy Function '''

compatability = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y))  # energy function

simulation = generate_simGrid_from_file(file, color=True, energy_function=compatability, grid_size=(8,8))

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