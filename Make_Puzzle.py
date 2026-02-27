import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
Setup
'''

def make_puzzle(file_in:str, file_out:str, rows = 8, columns=8):

    gray_matrix = cv2.imread(file_in, cv2.IMREAD_COLOR) # load in the file

    '''Chop up the image'''

    width = gray_matrix.shape[1] # horizontal distance - should be the shoter of the two
    length = gray_matrix.shape[0]
    tile_width = width//columns
    tile_length = length//rows

    tiles = []

    for i in range(rows):
        for j in range(columns):
            tiles.append(
                gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] # need this to reconstruct the array later
            )

    tiles = np.random.permutation(tiles)

    for i in range(rows):
        for j in range(columns):
            gray_matrix[tile_length*i:tile_length*(i+1),tile_width*j:tile_width*(j+1),:] = tiles[i*8 + j]


    cv2.imwrite(file_out,gray_matrix)

if __name__ == "__main__":
    make_puzzle("Inputs/"+"Original_Nebula.jpg","Inputs/"+"Nebula_Puzzle.jpg",40,60)
