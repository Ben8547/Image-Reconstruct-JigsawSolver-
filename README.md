# Usage Guide

## Project Purpose:
This project implements simulated annealing and genetic algorithm approaches to reconstructing a shuffled image grid (often reffered to a jigsaw puzzle in the literature) by minimizing an energy function defined over tile boundary compatibilities. Further in development I would like to add additional functionality.

## Quick Start Example
### Annealing
```python
from Annealing_Class import generate_simGrid_from_file, save_annealing_output

# Create simulation object from image
sim = generate_simGrid_from_file(
    filename="Inputs/Squirrel_Puzzle.jpg",
    grid_size=(8, 8),
    color=True,
    T0=10.0,
    Tf=0.5,
    geometric_decay_rate=0.9999
)

# Run annealing
sim.anneal()

# Save reconstructed image
save_annealing_output("Outputs/reconstructed.jpg", sim, color=True)
```
### Genetic Algorithm
```python
from Genome_Class import generate_genome_from_file, save_genome_output, genome_reconstruct

# Create Genome object from image
genome = generate_genome_from_file(
    filename="Inputs/Squirrel_Puzzle.jpg",
    grid_size=(8, 8),
    color=True,
    numberGenerations=100,
    parentsPerGeneration=4,
    populationSize=200,
    T0=10.0,
    Tf=0.5,
    geometric_decay_rate=0.9999,
    updates=True
)

# Run genetic algorithm
genome.run_simulation()

# Optional: run hybrid version with annealing refinement
# genome.run_simulation_with_annealing()

# Reconstruct final image
reconstructed = genome_reconstruct(genome, color=True)

# Save output
save_genome_output("Outputs/genetic_reconstruction.jpg", genome, color=True)
```

# Development Notes and Example Outputs

My take on the jigsaw problem.

The annealing algorithm has the capability to make 4 different moves. With 50% probability, it will swap two tiles (95% chance to choose a swap based on ideal compatibility within a random sample of tile), with 25% probability it will perform a roll operation on the representation matrix and with 25% probability it will swap two rectangular regions. To determine the efficacy of each of these operations consider the following example outputs each generated from the same puzzle image with an energy of 11512. The original squirrel has an energy of 7404. Each run presented below was completed with a geometric cooling schedule with initial temperature of 10 and a relaxation rate of 0.9999 and run until the temperature was 0.5 or less.

<figure>
  <img src="https://raw.githubusercontent.com/Ben8547/Image-Reconstruct-JigsawSolver-/main/ReadMeImages/pure_annealed_squirrel_single_swaps_only.jpg" width="300"/>
  <figcaption><em>Figure 1: Result of simulated annealing with single swaps only. The final energy was 9792.</em></figcaption>
</figure>
<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_subarray_only.jpg" width="300"/>
  <figcaption><em>Figure 2: Result of simulated annealing with rolls only. The final energy was 11447.</em></figcaption>
</figure>
<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_squirrel_rolls_only.jpg" width="300"/>
  <figcaption><em>Figure 3: Result of simulated annealing with subarray swaps only. The final energy was 11502.</em></figcaption>
</figure>
<figure>
  <img src="https://raw.githubusercontent.com/Ben8547/Image-Reconstruct-JigsawSolver-/main/ReadMeImages/pure_annealed_squirrel_with_rolls.jpg" width="300"/>
  <figcaption><em>Figure 4: Result of simulated annealing with single swaps and rolls. The final energy was 8763.</em></figcaption>
</figure>
<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_subarray_and_swaps.jpg" width="300"/>
  <figcaption><em>Figure 5: Result of simulated annealing with single swaps and subarray swaps The final energy was 8883.</em></figcaption>
</figure>
<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_all_three.jpg" width="300"/>
  <figcaption><em>Figure 6: Result of simulated annealing with all three movement options. The final energy was 8907.</em></figcaption>
</figure>


The above images are representative of several trials performed with each method. Notably, the variegation of movement options does tend to expedite solution discovery however simulated annealing alone seems incapable of finding the true solution on its own though it does dramatically simplify the puzzle for the human observer as it is efficient at grouping pieces. Single swaps alone only seem capable of putting together a single line of tiles while the addition of rolling operations allows for the conglomeration of these lines into larger structures. The subarray swaps and the rolls seemed to fill similar rolls in the process and I deemed them redundant. Since the subarray swaps seemed to slightly outperform rolls in accuracy I have opted to keep only subarray swaps. To further increase the efficacy of the algorithm we turn to a genetic algorithm described in https://doi.org/10.1109/CVPR.2013.231. While I do not implement their algorithm entirely, I use it's methods in combination with annealing to correct the absolute positions of tile.

Running the pure annealing algorithm with single and subarray swaps at 0.67 and 0.33 probability respectively with a geometric decay rate of 0.999999 and $$T_0 = 10$$ and $$T_f = 0.5$$ yield an approximate 3 hour run time and the following representative image which shows significant progress towards being solved.

<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_0.999999.jpg" width="300"/>
  <figcaption><em>Figure 6: Result of simulated annealing with very slow geometric scaling. The final energy was 7948.</em></figcaption>
</figure>

<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure-genome-color-10gens-100population.jpg" width="300"/>
  <figcaption><em>Figure 7: Result of genetic algorithm with 10 generations of 100 individuals per generation. The final energy was ____.</em></figcaption>
</figure>
