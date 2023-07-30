#!/usr/bin/env python
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

import numpy as np
from diffusion import *

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