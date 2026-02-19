import numpy as np
import cv2

'''Deicde color or non-color'''

color = True

'''Load in the test file (permanent)'''
file = "Original_Squirrel.jpg"
if color:
    color_volume = cv2.imread(file, cv2.IMREAD_COLOR)
    #print(color_volume)
    #print(color_volume.shape)
    '''
    This outputs a 3 dimensional array: (hight, width, 3)
    We will need to store data differently taking this into account.
    '''
else:
    gray_matrix = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #print(gray_matrix)

