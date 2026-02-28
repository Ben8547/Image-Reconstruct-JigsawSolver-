from Annealing_Class import generate_simGrid_from_file, save_annealing_output, annealing_reconstruct
from Genome_Class import generate_genome_from_file, save_genome_output, genome_reconstruct
import matplotlib.pyplot as plt
import numpy as np
from time import sleep, time
import cv2
from numba import njit

'''
Setup
'''

Color = True

file = "Inputs/"+"Original_Nebula.jpg"

@njit(fastmath=True, parallel=True)
def compatability(x,y):  # energy function
    return np.mean(np.abs(x-y))
#compatability = lambda x,y: np.mean(np.maximum(x,y) - np.minimum(x,y))

#simulation = generate_simGrid_from_file(file, color=Color, energy_function=compatability, grid_size=(20,30), T0=10., Tf=0.5, geometric_decay_rate=0.9999)
start_time = time()
simulation = generate_genome_from_file(file, color=Color,populationSize=100, numberGenerations=5, parentsPerGeneration=5, energy_function=compatability, grid_size=(20,20), T0=10., Tf=0.5, geometric_decay_rate=0.999, updates=True)

print(f"Initial Energy: {simulation.energy}")

#simulation.anneal()
simulation.run_simulation()
#simulation.run_simulation_with_annealing()
end_time = time()
print(f"Final energy {simulation.energy}")
print(f"Completed in {end_time-start_time} seconds")

restored_page = genome_reconstruct(simulation, color=Color)
#restored_page = annealing_reconstruct(simulation, color=Color)

#save_annealing_output("Outputs/"+"genome-color.jpg", simulation, Color, restored_page)
#save_genome_output("Outputs/"+"genome-color.jpg", simulation, Color, restored_page)

plt.imshow(restored_page)
plt.show()

'''Now that we have the ordered array, all that remains is to put the grayscale map back together.'''

