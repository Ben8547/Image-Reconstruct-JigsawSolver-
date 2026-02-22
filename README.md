My take on the jigsaw problem. **This is project is currently a work in progress**.

The annealing algorithm has the capability to make 4 different moves. With 50% probability, it will swap two tiles (95% chance to choose a swap based on ideal compatability within a random sample of tile), with 25% probabilty it will perform a roll operation on the representation matrix and with 25% probability it will swap two rectangular regions. To determine the efficacy of each of these operations consider the following example outputs each generated from the same puzzle image.

<figure>
  <img src="https://raw.githubusercontent.com/Ben8547/Image-Reconstruct-JigsawSolver-/main/ReadMeImages/pure_annealed_squirrel_single_swaps_only.jpg" width="300"/>
  <figcaption><em>Figure 1: Result of simulated annealing with single swaps only.</em></figcaption>
</figure>
<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_squirrel_rolls_only.jpg" width="300"/>
  <figcaption><em>Figure 1: Result of simulated annealing with rolls only.</em></figcaption>
</figure>
<figure>
  <img src="https://raw.githubusercontent.com/Ben8547/Image-Reconstruct-JigsawSolver-/main/ReadMeImages/pure_annealed_squirrel_with_rolls.jpg" width="300"/>
  <figcaption><em>Figure 2: Result of simulated annealing with single swaps and rolls.</em></figcaption>
</figure>

Each file represents an implementation of a distinct method.

* Make_Puzzle.py transforms and image into a scrambled puzzle.
* test-descrambler-simmulatedAnnealingOnly.py is my original method for puzzle solving, it uses a pure simulated annealing approach.
* test-genetic_algorithm_method.py is an adaptation of the method presented in https://doi.org/10.1109/CVPR.2013.231; I have added some annealing on top of their genetic algorithm.
* main.py will contian my most effecacious program once I am finished.
* Various included image files serve as test puzzles.
* Version3.py is a work in progress to vectorize the simmualted annealing model adding additional functionality as well.
* I have plans to implement some Gaussian process methods as well. I think that it would be interesting to compare the method to annealing.
