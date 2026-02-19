My take on the jigsaw problem. Here I use the example of a scrambled grayscale comicbook page.

Each file represents an implementation of a distinct method.

* Make_Puzzle.py transforms and image into a scrambled puzzle.
* test-descrambler-simmulatedAnnealingOnly.py is my original method for puzzle solving, it uses a pure simulated annealing approach.
* test-genetic_algorithm_method.py is an adaptation of the method presented in https://doi.org/10.1109/CVPR.2013.231; I have added some annealing on top of their genetic algorithm.
* main.py will contian my most effecacious program once I am finished.
