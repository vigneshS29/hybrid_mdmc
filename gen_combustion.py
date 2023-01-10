#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu 

# Imports
import numpy as np
import os, argparse, sys
from scipy.spatial.distance import *
from mol_classes import AtomList,IntraModeList
from lammps_files_classes import write_lammps_data,parse_data_header_and_masses

# Main argument.
def main(argv):

    # Create the argparse object
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument(dest='species', type=str,
                        help='Name of the species to add')

    parser.add_argument(dest='count', type=str,
                        help='Number of each species to add')

    # Optional arguments
    parser.add_argument('-prefix', dest='prefix', type=str, default='combustion',
                        help='Prefix of the resulting data file.')

    parser.add_argument('-msf', dest='msf', type=str, default='',
                        help='Name of the msf file.')

    parser.add_argument('-header', dest='header', type=str, default='',
                        help='Name of the data header file.')

    parser.add_argument('-output', dest='output', type=str, default='')

    # Parse the command line arguments.
    args = parser.parse_args()
    args.species = args.species.split()
    args.count = [ int(i) for i in args.count.split() ]
    if args.msf == '':
        args.msf = args.prefix+'.msf'
    if args.header == '':
        args.header = args.prefix+'.header'
    if args.output == '':
        args.output = args.prefix

    # Parse the msf and data header files
    msf = parse_msf(args.msf)
    header = parse_data_header_and_masses(args.header)
    type2mass = { i[0]:i[1] for i in header['masses'] }

    # Determine the minimum cube size that will fit all molecules to be placed
    cube = np.array([0,0,0])
    for sp in args.species:
        xyz = np.array([ i[4:] for i in msf[sp]['Atoms'] ])
        xyz = np.max(np.array([ np.max(xyz[:,i])-np.min(xyz[:,i]) for i in range(3) ]))
        xyz = np.array([ [xyz]*3 ])
        if np.linalg.norm(xyz) > np.linalg.norm(cube):
            cube = xyz
    if cube.all() == 0:
        cube = [[2,2,2]]

    # Determine the number of cubes needed and the box size needed
    num_cubes = 9
    while num_cubes < np.sum(args.count):
        num_cubes = num_cubes**3
    box = [ [-(round(num_cubes)**(1/3))*(cube[0][0])/2,(round(num_cubes)**(1/3))*(cube[0][0])/2] for i in range(3) ]

    # Create a list of center of geometries for the molecule placement
    cog = []
    for x in range(round(num_cubes**(1/3))):
        for y in range(round(num_cubes**(1/3))):
            for z in range(round(num_cubes**(1/3))):
                cog.append( np.array([ box[idx][0]+(i+1/2)*cube[0][idx] for idx,i in enumerate([x,y,z]) ]))
    np.random.shuffle(cog)

    # Create a shuffled list of molecule types to be placed
    to_place = [ item for sublist in [[args.species[idx]]*args.count[idx] for idx in range(len(args.species))] for item in sublist ]
    np.random.shuffle(to_place)

    # Place the molecules' atoms and interactions
    atoms,bonds,angles,dihedrals,impropers = AtomList(),IntraModeList(),IntraModeList(),IntraModeList(),IntraModeList()
    for sp_idx,sp in enumerate(to_place):
        num_atoms = len(atoms.ids)
        center = np.mean(np.array([ i[4:] for i in msf[sp]['Atoms'] ]), axis=0 )
        atoms.append(
            ids =         [ i[0]+num_atoms for i in msf[sp]['Atoms'] ],
            mol_id =      [ i[1]+len(set(atoms.mol_id)) for i in msf[sp]['Atoms'] ],
            lammps_type = [ i[2] for i in msf[sp]['Atoms'] ],
            charge =      [ i[3] for i in msf[sp]['Atoms'] ],
            x =           [ i[4]-center[0]+cog[sp_idx][0] for i in msf[sp]['Atoms'] ],
            y =           [ i[5]-center[1]+cog[sp_idx][1] for i in msf[sp]['Atoms'] ],
            z =           [ i[6]-center[2]+cog[sp_idx][2] for i in msf[sp]['Atoms'] ],
            mass =        [ type2mass[i[2]] for i in msf[sp]['Atoms'] ])
        if 'Bonds' in msf[sp].keys():
            bonds.append(
                ids =         [ i[0]+len(bonds.ids) for i in msf[sp]['Bonds'] ],
                lammps_type = [ i[1] for i in msf[sp]['Bonds'] ],
                atom_ids =    [ [j+num_atoms for j in i[2:]] for i in msf[sp]['Bonds'] ])
        if 'Angles' in msf[sp].keys():
            angles.append(
                ids =         [ i[0]+len(angles.ids) for i in msf[sp]['Angles'] ],
                lammps_type = [ i[1] for i in msf[sp]['Angles'] ],
                atom_ids =    [ [j+num_atoms for j in i[2:]] for i in msf[sp]['Angles'] ])
        if 'Dihedrals' in msf[sp].keys():
            dihedrals.append(
                ids =         [ i[0]+len(dihedrals.ids) for i in msf[sp]['Dihedrals'] ],
                lammps_type = [ i[1] for i in msf[sp]['Dihedrals'] ],
                atom_ids =    [ [j+num_atoms for j in i[2:]] for i in msf[sp]['Dihedrals'] ])
        if 'Impropers' in msf[sp].keys():
            impropers.append(
                ids =         [ i[0]+len(impropers.ids) for i in msf[sp]['impropers'] ],
                lammps_type = [ i[1] for i in msf[sp]['Impropers'] ],
                atom_ids =    [ [j+num_atoms for j in i[2:]] for i in msf[sp]['Impropers'] ])
    atoms.charge = atoms.charge-(np.sum(atoms.charge)/len(atoms.charge))

    # Write the data file
    data_header = { i:header[i] for i in ['masses','atom_types','bond_types','angle_types','dihedral_types','improper_types'] if i in header.keys() }
    write_lammps_data(args.output+'.in.data',atoms,bonds,angles,dihedrals,impropers,box,header=data_header)

    return


def parse_msf(msf_file):

    """Parser for master species file.

    Parameters
    ----------
    msf_file: str
        Name of the master species file.

    Returns Dictionary
    ------------------
    ~keys~
    Species: str

    ~values~
    'Atoms': list of lists
        [[atom_id,mol_id,atom_type,q,x,y,z], ...]

    'Bonds' or 'Angles' or 'Dihedrals' or 'Impropers': list of lists
        [[interaction_id,interaction_type,atom_i_id,atom_j_id, ...], ...]
    """

    msf = {}

    keywords = ['Atoms','Bonds','Angles','Dihedrals','Impropers']
    flag = None

    with open(msf_file,'r') as f:
        for line in f:
            fields = line.split()
            if fields == []: continue
            if fields[0] == '#': continue
            if fields[0] == 'Species:':
                sp = fields[1]
                msf[sp] = {}
                continue
            if fields[0] in keywords:
                flag = fields[0]
                continue
            if flag == 'Atoms':
                if flag not in msf[sp].keys(): msf[sp][flag] = []
                msf[sp]['Atoms'].append([int(i) for i in fields[:3]]+[float(i) for i in fields[3:]])
                continue
            if flag:
                if flag not in msf[sp].keys(): msf[sp][flag] = []
                msf[sp][flag].append([int(i) for i in fields])

    return msf

if __name__ == '__main__':
    main(sys.argv[1:])
