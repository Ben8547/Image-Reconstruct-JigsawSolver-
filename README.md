My take on the jigsaw problem. **This is project is currently a work in progress**.

The annealing algorithm has the capability to make 4 different moves. With 50% probability, it will swap two tiles (95% chance to choose a swap based on ideal compatability within a random sample of tile), with 25% probabilty it will perform a roll operation on the representation matrix and with 25% probability it will swap two rectangular regions. To determine the efficacy of each of these operations consider the following example outputs each generated from the same puzzle image with an energy of 11512. The original squirrel has an energy of 7404. Each run presented below was completed with a geometric cooling schedule with initial tempurature of 10 and a relatxation rate of 0.999 and run until the tempurature was 0.5 or less.

<figure>
  <img src="https://raw.githubusercontent.com/Ben8547/Image-Reconstruct-JigsawSolver-/main/ReadMeImages/pure_annealed_squirrel_single_swaps_only.jpg" width="300"/>
  <figcaption><em>Figure 1: Result of simulated annealing with single swaps only. The final energy was 9792.</em></figcaption>
</figure>
<figure>
  <img src="https://github.com/Ben8547/Image-Reconstruct-JigsawSolver-/blob/main/ReadMeImages/pure_annealed_squirrel_rolls_only.jpg" width="300"/>
  <figcaption><em>Figure 1: Result of simulated annealing with rolls only. The final energy was 11447.</em></figcaption>
</figure>
<figure>
  <img src="https://raw.githubusercontent.com/Ben8547/Image-Reconstruct-JigsawSolver-/main/ReadMeImages/pure_annealed_squirrel_with_rolls.jpg" width="300"/>
  <figcaption><em>Figure 2: Result of simulated annealing with single swaps and rolls. The final energy was 8763.</em></figcaption>
</figure>

The above images are representative of several trials performed with each method. Noteably, the variagation of movement options does tend to expidite solution discovery however simmulated annealing alone seems incapable of finding the true solution on its own though it does dramatically simplify the puzzle for the human observer as it is efficient at grouping pieces. Single swaps alone only seem capable of putting together a single line of tiles while the addition of rolling operations allows for the conglomeration of these lines into larger structures. To further increase the efficacy of the algorithm we turn to a genetic algorithm descirbed in https://doi.org/10.1109/CVPR.2013.231. While I do not implement their algorithm entirely, I use it's methods in combination with annealing to correct the abolsoute positions of tile.

Each file represents an implementation of a distinct method.

* Make_Puzzle.py transforms and image into a scrambled puzzle.
* test-descrambler-simmulatedAnnealingOnly.py is my original method for puzzle solving, it uses a pure simulated annealing approach.
* test-genetic_algorithm_method.py is an adaptation of the method presented in https://doi.org/10.1109/CVPR.2013.231; I have added some annealing on top of their genetic algorithm.
* main.py will contian my most effecacious program once I am finished.
* Various included image files serve as test puzzles.
* Version3.py is a work in progress to vectorize the simmualted annealing model adding additional functionality as well.
* I have plans to implement some Gaussian process methods as well. I think that it would be interesting to compare the method to annealing.
