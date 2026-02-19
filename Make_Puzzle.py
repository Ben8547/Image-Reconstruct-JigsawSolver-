import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
Setup
'''

gray_matrix = cv2.imread("Original_Squirrel.jpg", cv2.IMREAD_GRAYSCALE) # load in the file

'''Chop up the image'''

width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
length = gray_matrix.shape[0]
tile_width = width//8
tile_length = length//8

tiles = []

for i in range(8):
    for j in range(8):
        tiles.append(
            gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] # need this to reconstruct the array later
        )

tiles = np.random.permutation(tiles)

for i in range(8):
    for j in range(8):
        gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] = tiles[i*8 + j]


cv2.imwrite(f"Squirrel_Puzzle.jpg",gray_matrix)
