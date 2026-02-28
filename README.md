# Table of Contents

- [Project Purpose](#project-purpose)
- [Dependancies](#dependancies)
- [Preliminary Examples](#preliminary-examples)
  - [Annealing](#annealing)
  - [Genetic Algorithm](#genetic-algorithm)
- [Annealing Documentation](#annealing-documentation)
  - [simulation_grid](#annealing_classsimulation_grid)
  - [anneal](#annealing_classsimulation_gridanneal)
  - [Cooling Schedule (Geometric)](#annealing_classsimulation_gridcooling_schedule_geometric)
  - [Cooling Schedule (Optimal)](#annealing_classsimulation_gridcooling_schedule_optimal)
  - [generate_simGrid_from_file](#annealing_classgenerate_simgrid_from_file)
  - [annealing_reconstruct](#annealing_classannealing_reconstruct)
  - [save_annealing_output](#annealing_classsave_annealing_output)
- [Genetic Algorithm Documentation](#genetic-algorithm-documentation)
  - [Genome](#genome_classgenome)
  - [run_simulation](#genome_classgenomerun_simulation)
  - [run_simulation_with_annealing](#genome_classgenomerun_simulation_with_annealing)
  - [generate_genome_from_file](#genome_classgenerate_genome_from_file)
  - [genome_reconstruct](#genome_classgenome_reconstruct)
  - [save_genome_output](#genome_classsave_genome_output)
- [Development Notes and Example Outputs](#development-notes-and-example-outputs)
  - [Benchmarking](#benchmarking)
  	-[Bechmarks for Energy Caching](#bechmarks-for-energy-caching)   

# Project Purpose:
This project implements simulated annealing and genetic algorithm approaches to reconstructing a shuffled image grid (often reffered to a jigsaw puzzle in the literature) by minimizing an energy function defined over tile boundary compatibilities. Further in development I would like to add additional functionality.

## Dependancies

```cmd
pip install numba, numpy, cv2
```

## Preliminary Examples
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

# Annealing Documentation

## ```Annealing_Class.simulation_grid```
Main simulation object used to perform annealing.
# Attributes

```simGrid : ndarray```
Current tile arrangement (indices referencing ```tile_data```).

```grid_shape : tuple[int, int]```
The dimensions of ```simGrid```.

```tile_data : ndarray(dtype=object)```
Array of storing tile boundary slices and full tile image.

``` cached_energies : ndarray ```
Precomputed compatibility energies between tile edges.

```energy : float```
Current total energy of ```simGrid```.

``` T0 : float ```
Initial simulation tempurature.

```Tf : float```
Final temperature; the simulation halts at this tempurature.

```geometric_rate : float```
Tempurature schudle definiging constant. The class uses ```T(k) = geometric_rate**k * T0 ``` to determine the tempurature of iteration ```k```.

### Notes
* All pairwise boundary energies are cached in a 3D array of shape: ```(num_tiles, num_tiles, 4)```
* Direction indexing is given by:
    * ```0```$\to$ top
    * ```1```$\to$ left
    * ```2```$\to$ bottom
    * ```3```$\to$ right

## ```Annealing_Class.simulation_grid.anneal```
```simulation_grid.anneal()```
Run the simulated annealing process by Iteratively performing:
<ol>
    <li>A single Markov step</li>
    <li>Energy revaluation</li>
    <li>Metropolis acceptance rule</li>
    <li>Temperature update</li>
</ol>
The temperature is updated using geometric cooling ``'simulation_grid.cooling_schedule_geometric``` and the algorithm terminates when ```T <= Tf```.

The Markov step can make one of three perturbations to the grid:
<ol>
    <li>With probability $0.6\overline{3}$: Single-tile swaps biased towards tiles with high local energy.</li>
    <li>With probability $0.0\overline{3}$: Random Single-tile swaps.</li>
    <li>With probability $0.\overline{3}$: Subarray swaps.</li>
</ol>

## ```Annealing_Class.simulation_grid.cooling_schedule_geometric```
```cooling_schedule_geometric(T_0, rate, k)```
Geometric cooling schedule. Computes ``` T(k) = T0 * rate**k ```
Thus the program reaches ```Tf``` when $k = \frac{\ln\left(\frac{T_f}{T_0}\right)}{\ln(r)}$ and hence scale relatively moderately with the rate. When the rate is near $1$, linear approximation yields that for $r = 1-\varepsilon$, $k\approx \frac{\ln\left(\frac{T_0}{T_f}\right)}{\varepsilon}$. Hence changing the ration of tempuratures has a negligiable impact on the completion time of the algorithm compared to logarithmic cooling.

# Parameters
```T_0 : float```
Initial temperature.

```rate : float```
Multiplicative decay factor.

```k : int```
Iteration index.

# Returns
```float```
Updated temperature.

## ```Annealing_Class.simulation_grid.cooling_schedule_optimal```
```cooling_schedule_optimal(T_0, k)```
A logarithmic cooling schedule which is proven to lead to convergence of the Markov chain given some additional assumptions by Hajek in 1987.
Tempurature is computed according to ```T(k) = T0 / log(k)```. We find that the number of iterations required to complete the simulations is $k = e^{T_0/T_f}$ and thus depends strongly on the ratio of the tempuratures. Since we require $T_f\to0$ for convergence it is clear that this algorithm would take infinite time. The geometric cooling schedule is preffered by this program for this reason and this function is not used by ```simulation_grid.anneal()```.


## ```Annealing_Class.generate_simGrid_from_file```
```python
generate_simGrid_from_file(
    filename = "Inputs/Squirrel_Puzzle.jpg",
    grid_size = (8,8),
    color = True,
    energy_function = lambda x,y: np.mean(np.maximum(x,y)-np.minimum(x,y)),
    T0=10.,
    Tf=0.5,
    geometric_decay_rate = 0.9999
) -> simulation_grid ```
```
Create a simulation_grid object directly from an image file. The image is partitioned into rectangular tiles. Boundary energies between every pair of tiles are precomputed and cached.

### Parameters
``` filename : str ```
Path to input image file. I've only tested this with .jpg files.

```grid_size : tuple[int, int]```
Shape ```(rows, cols)``` specifying how many tiles the image is split into.

```color : bool```
If True, the image is processed in RGB. If False, the image is processed in grayscale.

``` energy_function : callable ```
Function used to compute compatibility between tile boundaries. It must be able accept two NumPy arrays and return a scalar energy value. Remember that this function will be minimized by the annealing process.

``` T0 : float ```
Initial annealing temperature.

```Tf : float```
Final annealing temperature - the algorithm halts when the process reaches this tempurature.

```geometric_decay_rate : float```
Geometric cooling multiplier (must be < 1 or else the program will never halt).

### Returns
```simulation_grid```
An initialized annealing simulation object.

## ```Annealing_Class.annealing_reconstruct```
```annealing_reconstruct(simulation, color=True)```
Reconstruct the full image from the current tile configuration.
### Parameters
```simulation : simulation_grid```
Simulation object.

```color : bool```
Determines whether image is RGB or grayscale.

### Returns
```ndarray```
Reconstructs the image as an ```uint8``` array.

## ```Annealing_Class.save_annealing_output```
```save_annealing_output(filename, simulation, color=True)```
Save the reconstructed image to a file.
### Parameters
```filename : str```
Output file path.

```simulation : simulation_grid```
Simulation object.

```color : bool```
Determines whether image is RGB or grayscale.

# Genetic Algorithm Documentation

## ```Genome_Class.Genome```
Main object implementing the genetic algorithm (with optional annealing refinement) for puzzle reconstruction. The genetic algorithm evolves a population of candidate grids by minimizing the same boundary-based energy function used in the annealing class. This is a functional implementation of the algorithm discussed by Sholomon et al. here: [https://doi.org/10.48550/arXiv.1711.06769](url).

# Attributes

```population : list[ndarray]```
List of candidate grids (each grid is an index matrix referencing tile_data).

```population_energies : list[float]```
Energy value associated with each grid in the population.

```tile_data : ndarray(dtype=object)```
Array storing tile boundary slices and full tile image.

```cached_energies : ndarray```
Precomputed compatibility energies between tile edges.

```grid_shape : tuple[int, int]```
Shape of each chromosome grid.

```numberGenerations : int```
Total number of generations to evolve.

```parentsPerGeneration : int```
Number of lowest-energy individuals selected as parents.

```populationSize : int```
Total number of individuals in each generation.

```T0, Tf, geometric_rate : float```
Annealing parameters used if hybrid annealing refinement is enabled. See [Annealing Documentation](#annealing-documentation) for more information.

```
product : ndarray
```
Lowest-energy grid found during evolution.

```energy : float```
Energy of the best genome found at each iteration.

```updates : bool```
If True, prints generation progress and energy updates.

## ```Genome_Class.Genome.run_simulation```
```Genome.run_simulation()```
Execute the genetic algorithm.

Each generation performs:

<ol>
  <li>Energy evaluation of entire population</li> <li>Selection of top ```parentsPerGeneration``` individuals</li>
  <li>Crossover to produce offspring</li> <li>Mutation to introduce variation</li>
  <li>Population replacement</li>
</ol>

The algorithm halts after ```Genome.numberGenerations``` iterations and the individual with minimum energy across all generations is stored in: ```Genome.product```.

## ```Genome_Class.Genome.run_simulation_with_annealing```

```Genome.run_simulation_with_annealing()```
This runs the same genetic algorithm as ```Genome.run_simulation``` except applies annealing to each of the initial parents in order to find initial tile-pairing. There is a final annealing phase for the lowest energy result. For the images I tested, this was slower than ```Genome.run_annealing()``` with no added benifit. It may be desireable for larger grids.

## Genome_Class.generate_genome_from_file
```python
generate_genome_from_file(
    filename="Inputs/Squirrel_Puzzle.jpg",
    grid_size=(8,8),
    color=True,
    numberGenerations=100,
    parentsPerGeneration=4,
    populationSize=200,
    T0=10.,
    Tf=0.5,
    geometric_decay_rate=0.9999,
    updates=True
) -> Genome
```
Create a ```Genome``` object directly from an image file. The image is partitioned into rectangular tiles.
Boundary energies between every pair of tiles are precomputed and cached.

# Parameters

```filename : str```
Path to input image file.

```grid_size : tuple[int, int]```
Shape ```(rows, cols)``` specifying how many tiles the image is split into.

```color : bool```
If True, image is processed in RGB. If False, image is processed in grayscale.

```numberGenerations : int```
Number of generations to evolve.

```parentsPerGeneration : int```
Number of energy-miniized individuals selected each generation.

```populationSize : int```
Total number of chromosome grids per generation.

```T0, Tf, geometric_decay_rate : float```
Annealing parameters used if hybrid mode is enabled. See [Annealing Documentation](#annealing-documentation) for more information.

```updates : bool```
Print progress updates during evolution.

# Returns
```Genome```
Initialized genetic algorithm object.

## ```Genome_Class.genome_reconstruct```
```genome_reconstruct(genome, color=True)```
Reconstruct the full image from the best genome configuration.

# Parameters
```genome : Genome```
A ```Genome``` object.

```color : bool```
Determines whether image is RGB or grayscale.

# Returns
```ndarray```
Reconstructed image as an ```uint8``` array.

## ```Genome_Class.save_genome_output```
```save_genome_output(filename, genome, color=True)```
Save the reconstructed image to a file.

# Parameters
```filename : str```
Output file path.

```genome : Genome```
Genome object.

```color : bool```
Determines whether image is RGB or grayscale.

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

## Benchmarking
### Bechmarks for Energy Caching:
* For a 20x20 puzzle (Original_Nebula.jpg) we get:
	* No NUMBA: 8.893833875656128 seconds
	* NUMBA with compile time: 15.388114213943481 seconds
	* NUMBA without compile time: 6.202281475067139 seconds
* For a 40x60 puzzle (Original_Nebula.jpg) we get:
	* No NUMBA: 293.23735904693604 seconds
	* NUMBA with compile time: 231.85261917114258 seconds
	* NUMBA without compile time: 215.7313826084137 seconds
