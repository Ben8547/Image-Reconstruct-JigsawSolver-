from Annealing_Class import generate_simGrid_from_file, save_annealing_output, annealing_reconstruct
from Genome_Class import generate_genome_from_file, save_genome_output, genome_reconstruct
from Bayes_Class import generate_bayes_from_file, save_bayes_output, bayes_reconstruct
import matplotlib.pyplot as plt
import numpy as np
from time import sleep, time
import cv2
from numba import njit

'''
Setup
'''

@njit(fastmath=True, parallel=True)
def compatability(x,y):  # energy function
    return np.mean(np.abs(x-y))
#compatability = lambda x,y: np.mean(np.maximum(x,y) - np.minimum(x,y))

solver = "Bayes" # may be SA, GA, Linear, or Bayes

Color = True

file = "Inputs/"+"Squirrel_Puzzle.jpg"

puzzle_size = (8,8)


start_time = time()
if solver == "GA":
    simulation = generate_genome_from_file(file, color=Color,populationSize=100, numberGenerations=5, parentsPerGeneration=5, energy_function=compatability, grid_size=puzzle_size, T0=10., Tf=0.5, geometric_decay_rate=0.999, updates=True)
    print(f"Initial Energy: {simulation.energy}")
    simulation.run_simulation()
    #simulation.run_simulation_with_annealing()
    end_time = time()
    print(f"Final energy {simulation.energy}")
    print(f"Completed in {end_time-start_time} seconds")
    restored_page = genome_reconstruct(simulation, color=Color)
    #save_genome_output("Outputs/"+"genome-color.jpg", simulation, Color, restored_page)
elif solver == "SA":
    simulation = generate_simGrid_from_file(file, color=Color, energy_function=compatability, grid_size=puzzle_size, T0=10., Tf=0.5, geometric_decay_rate=0.9999)
    print(f"Initial Energy: {simulation.energy}")
    simulation.anneal()
    end_time = time()
    print(f"Final energy {simulation.energy}")
    print(f"Completed in {end_time-start_time} seconds")
    restored_page = annealing_reconstruct(simulation, color=Color)
    #save_annealing_output("Outputs/"+"genome-color.jpg", simulation, Color, restored_page)
elif solver == "Linear":
    pass
elif solver == "Bayes":
    simulation = generate_bayes_from_file(file, color=Color, energy_function=compatability, init_points=100, n_iter=250)
    end_time = time()
    print(f"Completed in {end_time-start_time} seconds")
    print(f"Final energy {simulation.energy}")
    restored_page = bayes_reconstruct(simulation, color=Color)
    #save_bayes_output("Outputs/"+"genome-color.jpg", simulation, Color, restored_page)
else:
    raise ValueError("solver is not properly specified")



restored_page = np.roll(restored_page,1,axis=2) # cv2 color orders do not natively match the matplotlib orders; this fixes that so that we can view the colors correctly; converts BGR in cv2 to RGB
plt.imshow(restored_page)
plt.show()

