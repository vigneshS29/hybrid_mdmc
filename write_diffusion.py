#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu 

# Imports
import numpy as np
import os, argparse, sys, datetime
import pipes, json
from scipy.spatial.distance import *
from copy import deepcopy
from mol_classes import AtomList,IntraModeList
from hybrid_mdmc.classes import MoleculeList,ReactionList
from data_file_parser import parse_data_file
from lammps_files_classes import write_lammps_data,write_lammps_init

# Main argument.
def main(argv):    
    """Script for writing the diffusion file used in hybrid MD/MC.

    Dependencies
    ------------
    mol_classes: AtomList,IntraModeList
    data_file_parser: parse_data_file
    frame_generator

    Required Files
    --------------
    data_file:
        LAMMPS data file, used for retrieving atom position and type data.
    trajectory_file:
        Lammps trajectory file.
    """

    # Create the argparse object
    parser = argparse.ArgumentParser(description='Creates diffusion file.')

    # Positional arguments
    parser.add_argument(dest='data_file', type=str,
                        help='Name of the LAMMPS data file to read in.')

    #parser.add_argument(dest='trajectory_file', type=str,
    #                    help='Name of the trajectory file to read in.')

    # Optional arguments
    parser.add_argument('-prefix', dest='prefix', type=str, default='default',
                        help='Prefix of the rxndf file, msf file, and all output files. Prefix can be overridden for individual files. Default: data_file prefix')

    parser.add_argument('-temp', dest='temp', type=float, default=298.0,
                        help='Temperature of the simulation (K). Default: 298.0 K')

    parser.add_argument('-num_voxels', dest='num_voxels', type=str, default='3 3 3',
                        help='Space delimited strong of the number of voxels in the x, y, and z directions. Default: 3 3 3')

    parser.add_argument('-x_bounds', dest='x_bounds', type=str, default='',
                        help='Space delimited strong defining custom bounds for the x-axis. Overrides first entry in -num_voxels')

    parser.add_argument('-y_bounds', dest='y_bounds', type=str, default='',
                        help='Space delimited strong defining custom bounds for the y-axis. Overrides second entry in -num_voxels')

    parser.add_argument('-z_bounds', dest='z_bounds', type=str, default='',
                        help='Space delimited strong defining custom bounds for the z-axis. Overrides third entry in -num_voxels')
    
    parser.add_argument('--distance_criteria', dest='distance_criteria', default=False, action='store_const', const=True)
    
    # Parse the command line arguments.
    args = parser.parse_args()
    if args.prefix=='default':
        args.prefix = args.data_file
        if args.data_file.endswith('.in.data'):
            args.prefix = args.data_file[:-8]
        elif args.data_file.endswith('.end.data'):
            args.prefix = args.data_file[:-9]
        elif args.data_file.endswith('.data'):
            args.prefix = args.data_file[:-5]

    # Parse the data_file, diffusion_file, rxndf, and msf files
    atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(args.data_file,unwrap=True)
    box2 = [ [np.min(atoms.x),np.max(atoms.x)], [np.min(atoms.y),np.max(atoms.y)], [np.min(atoms.z),np.max(atoms.z)] ]
    for idx in range(3):
        if box[idx][0] > box2[idx][0]:
            box[idx][0] = box2[idx][0]
        if box[idx][1] < box2[idx][1]:
            box[idx][1] = box2[idx][1]
    
    # Define the voxels
    voxel_bounds = []
    given_bounds = [ [float(i) for i in args.x_bounds.split()], [float(i) for i in args.y_bounds.split()], [float(i) for i in args.z_bounds.split()] ]
    for d in range(3):
        if len(given_bounds[d]):
            voxel_bounds.append(given_bounds[d])
            continue
        voxel_bounds.append([ np.min(box[d])+(np.max(box[d])-np.min(box[d]))/int(args.num_voxels.split()[d])*i for i in range(int(args.num_voxels.split()[d])) ])
        voxel_bounds[-1].append(np.max(box[d]))
    voxels = {}
    count = 1
    for i in range(len(voxel_bounds[0])-1):
        for j in range(len(voxel_bounds[1])-1):
            for k in range(len(voxel_bounds[2])-1):
                idx = [i,j,k]
                voxels[count] = {'bounds':[ [voxel_bounds[d][idx[d]],voxel_bounds[d][idx[d]+1]] for d in range(3) ]}
                count += 1

    # Calculate the diffusion matrix
    if args.distance_criteria:
        geo = np.array([ [np.mean(voxels[v]['bounds'][d]) for d in range(3)] for v in sorted(voxels.keys()) ])
        diffusion_matrix = 1-find_scaled_dist_same(geo,box)

    # Write the diffusion file
    with open(args.prefix+'.diffusion','w') as f:
        f.write('# Diffusion file for {}\n'.format(argv[0]))
        f.write('# Generated: {}\n'.format(datetime.datetime.now()))
        f.write('# Arguments: {}\n\n'.format(str(sys.argv[1:])))
        for v in sorted(list(voxels.keys())):
            f.write('voxel {}\n'.format(v))
            f.write('  xbounds {:>14.8f} {:>14.8f}\n'.format(voxels[v]['bounds'][0][0],voxels[v]['bounds'][0][1]))
            f.write('  ybounds {:>14.8f} {:>14.8f}\n'.format(voxels[v]['bounds'][1][0],voxels[v]['bounds'][1][1]))
            f.write('  zbounds {:>14.8f} {:>14.8f}\n\n'.format(voxels[v]['bounds'][2][0],voxels[v]['bounds'][2][1]))
        f.write('diffusion_matrix\n')
        for row in diffusion_matrix:
            f.write('  {}\n'.format(' '.join(['{:>6.4f}'.format(i) for i in row])))
    return

# Computes the pairwise distance between particles in a list, scaled by the box dimensions
def find_scaled_dist_same(geo,box):
    rs = np.zeros((geo.shape[0], geo.shape[0]))
    for i in range(3):
        dist = squareform(pdist(geo[:,i:i+1], 'minkowski', p=1.0))
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all():
            dist -= l * (dist > l2)
            dist = np.abs(dist)
        rs += (dist/(box[i][1]-box[i][0]))**2
    rs = np.sqrt(rs)
    return rs

if __name__ == '__main__':
    main(sys.argv[1:])
