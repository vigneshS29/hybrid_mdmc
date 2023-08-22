#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

import numpy as np


# start by running everything with non-development functions, etc.
# if they crash, look at difference bewteen non-development and development functions, etc.
# then assimilate

#%% System

# Species: A, A2, A2B, B

# Ten of each molecule, 40 total molecules, 60 total atoms
# 27 total voxels
# Ten frames

# Required files:
# .lammpstrj
# .end.data
# .msf


#%% Test diffusion.py

transitions_matrix = np.array([
    [ 0, 10,  4,  0,  0,  0,  9],
    [ 7,  0,  8,  0,  0,  0, 20],
    [ 0,  6,  0,  5,  0,  2,  0],
    [ 0,  0,  9,  0,  8,  0,  0],
    [ 0,  0,  0,  9,  0,  9,  0],
    [ 0,  0,  0,  0, 10,  0,  7],
    [10,  0,  0,  0,  0,  8,  0],
])
dt = 10
transitionrate_matrix = transitions_matrix/dt

diffusionrate_matrix = get_diffusionratematrix_from_transitionratematrix(transitionrate_matrix)