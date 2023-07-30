#!/usr/bin/env python3
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu


import numpy as np
import pandas as pd
from hybrid_mdmc.Development.calc_voxels import *
from hybrid_mdmc.Development.functions import get_voxel_neighbors_byID

num_voxels = [3,3,3]
box = [[-12,12],[-9,9],[-6,6]]
box = [[-12,12],[-12,12],[-12,12]]
voxels = calc_voxels(num_voxels,box)
breakpoint()

voxel_neighbors_byID = get_voxel_neighbors_byID(voxels)
breakpoint()
