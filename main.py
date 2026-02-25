from Version3 import generate_simGrid_from_file, save_output, reconstruct
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import cv2

'''
Setup
'''

Color = True

file = "Inputs/"+"Squirrel_Puzzle.jpg"

compatability = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y))  # energy function

simulation = generate_simGrid_from_file(file, color=Color, energy_function=compatability, grid_size=(8,8))

print(f"Initial Energy: {simulation.energy}")

simulation.anneal()

print(f"Final energy {simulation.energy}")

restored_page = reconstruct(simulation, color=Color)

save_output("Outputs/"+"annealing-color.jpg", simulation)

plt.imshow(restored_page)
plt.show()

'''Now that we have the ordered array, all that remains is to put the grayscale map back together.'''

