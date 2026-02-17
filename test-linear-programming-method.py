import numpy as np
import matplotlib.pyplot as plt
import cv2

gray_matrix = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE) # load in the file

'''Chop up the image'''

width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
length = gray_matrix.shape[0]
tile_width = width//8
tile_length = length//8

tiles = []

for i in range(8):
    for j in range(8):
        tiles.append({
            "top": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j],
            "bottom": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*(j+1)-1],
            "left": gray_matrix[tile_length*i,tile_width*j:tile_width*(j+1)],
            "right": gray_matrix[tile_length*(i+1)-1,tile_width*j:tile_width*(j+1)],
            "entire": gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1)] # need this last one to reconstruct the array later
        })

del gray_matrix

