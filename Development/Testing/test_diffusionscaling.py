#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

# %% Imports
import pprint
import numpy as np
from hybrid_mdmc.parsers import *
from hybrid_mdmc.functions import *
from hybrid_mdmc.data_file_parser import parse_data_file

# start by running everything with non-development functions, etc.
# if they crash, look at difference bewteen non-development and development functions, etc.
# then assimilate

# System

# Species: A, A2, A2B, B

# Ten of each molecule, 40 total molecules, 60 total atoms
# 27 total voxels
# Ten frames

# Required files:
# .lammpstrj
# .end.data
# .msf


# %% Test hybrid_mdmc.Development.calc_voxels
from hybrid_mdmc.Development.calc_voxels import *
print('Testing hybrid_mdmc.Development.calc_voxels...')

# Create requested number of voxels, boxes, and voxel bounds
num_voxels = [3,3,3]
box = [[-12.0,12.0],[-24.0,24.0],[-12.0,12.0]]
givenbounds = [-12.0,0.0,1.0,2.0,12.0]

# Create voxels
voxels = [
    calc_voxels(num_voxels,box),
    calc_voxels(num_voxels,box,xbounds=givenbounds),
]
pprint.pprint(voxels)

# Recover box from the voxels
box_ = [
    get_box_fromvoxels(voxels[0]),
    get_box_fromvoxels(voxels[1]),
]
pprint.pprint(box_)

# Attempt to raise an error
voxels = [
    calc_voxels([],box,xbounds=givenbounds,ybounds=givenbounds,zbounds=givenbounds)
]

# %% Test hybrid_mdmc.Development.diffusion.py
from hybrid_mdmc.Development.diffusion import *

# Create a DiffusionGraph object with fictious transition data
print('testing "DiffusionGraph" class...')
error = False
transition_matrix = np.array([
    [ 0, 10,  4,  0,  0,  0,  9],
    [ 7,  0,  8,  0,  0,  0, 20],
    [ 0,  6,  0,  5,  0,  2,  0],
    [ 0,  0,  9,  0,  8,  0,  0],
    [ 0,  0,  0,  9,  0,  9,  0],
    [ 0,  0,  0,  0,  0,  0,  0],
    [10,  0,  0,  0,  0,  8,  0],
])
dt = 10
transitionrate_matrix = transition_matrix/dt
transitionrate_DiffusionGraph = DiffusionGraph(
    edges={row+1:[_+1 for _,val in enumerate(transitionrate_matrix[row]) if val ]
           for row in range(len(transitionrate_matrix))},
    weights={(rowidx+1,colidx+1):transitionrate_matrix[rowidx,colidx]
             for rowidx in range(len(transitionrate_matrix))
             for colidx in range(len(transitionrate_matrix)) if transitionrate_matrix[rowidx,colidx]}
)
for rowidx in range(len(transition_matrix)):
    if rowidx+1 not in transitionrate_DiffusionGraph.edges.keys():
        error = True
        print('    error: not all voxel IDs are keys in edges')
    for colidx in range(len(transition_matrix)):
        if transition_matrix[rowidx,colidx]:
            if colidx+1 not in transitionrate_DiffusionGraph.edges[rowidx+1]:
                error = True
                print('    error: not all all connceted voxel IDs are included in edges lists')
        else:
            if colidx+1 in transitionrate_DiffusionGraph.edges[rowidx+1]:
                error = True
                print('    error: extraneous voxel IDs in edge lists')
    correct_weight_keys = [colidx+1 for colidx,_ in enumerate(transition_matrix[rowidx]) if _]
    actual_weight_keys = [_[1] for _ in transitionrate_DiffusionGraph.weights.keys() if _[0] == rowidx+1]
    # check all weight keys are present
    if not all(_ in actual_weight_keys for _ in correct_weight_keys):
        error = True
        print('    error: not all weights are included')
    # check no exgra weight keys are present
    if not all(_ in correct_weight_keys for _ in actual_weight_keys):
        error = True
        print('    error: extraneous weights included')
    # check corret weights
    if not all(transition_matrix[rowidx,colidx]/dt == transitionrate_DiffusionGraph.weights[(rowidx+1,colidx+1)]
               for colidx in range(len(transition_matrix)) if transition_matrix[rowidx,colidx]):
        error = True
        print('    error: incorrect weights calculated')
if not error:
    print('    --> successful')

print('testing "get_DiffusionGraph_from_matrix" function...')
error = False
transitionrate_DiffusionGraph1 = get_DiffusionGraph_from_matrix(transitionrate_matrix)
if not transitionrate_DiffusionGraph.edges == transitionrate_DiffusionGraph1.edges:
    error = True
    print('    error: edges calculated incorrectly')
if not transitionrate_DiffusionGraph.weights == transitionrate_DiffusionGraph1.weights:
    error = True
    print('    error: weights calculated incorrectly')
if not error:
    print('    --> successful')

# %% Test dijkstra

matrix = np.array([
    [0,1,9,9,0,0,0,0,0,0], #0
    [1,0,9,0,1,9,0,0,0,0], #1
    [9,9,0,9,0,1,1,9,0,0], #2
    [9,0,9,0,0,0,0,9,9,0], #3
    [0,1,0,0,0,1,0,0,0,0], #4
    [0,9,1,0,1,0,0,0,0,0], #5
    [0,0,1,0,0,0,0,0,0,1], #6
    [0,0,9,9,0,0,0,0,0,9], #7
    [0,0,0,9,0,0,0,0,0,0], #8
    [0,0,0,0,0,0,1,9,0,0], #9
])
DG = get_DiffusionGraph_from_matrix(matrix)
path,tw = dijkstra(DG,0,9)
print(path)
print(tw)


# %% Test get_voxels_byframe
trjfile = 'test_diffusionscaling.production.lammpstrj'
datafile = 'test_diffusionscaling.in.data'
msffile = 'test_diffusionscaling.msf'
atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(datafile,atom_style='molecular')
masterspecies = parse_msf(msffile)
num_voxels = [3,3,3]
voxels = calc_voxels(num_voxels,box)
voxelsmap,voxelsx,voxelsy,voxelsz = voxels2voxelsmap(voxels)
voxelID2idx = {k:idx for idx,k in enumerate(sorted(list(voxels.keys())))}
atomtypes2moltype = {}
for k,v in masterspecies.items():
    atomtypes2moltype[tuple(sorted([i[2] for i in v['Atoms']]))] = k
dfmolecules = gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz)
voxels_byframe,voxelID2idx = get_voxels_byframe(trjfile,atoms,dfmolecules,num_voxels)




# %% Test calc_diffusionrate
trjfile = 'test_diffusionscaling.production.lammpstrj'
datafile = 'test_diffusionscaling.in.data'
msffile = 'test_diffusionscaling.msf'
atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(datafile,atom_style='molecular')
masterspecies = parse_msf(msffile)
diffusion_rate = calc_diffusionrate(
    trjfile,atoms,box,masterspecies,[6,6,6])
# %%
