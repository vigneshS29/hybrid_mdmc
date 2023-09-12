#!/usr/bin/env python3
# Author
#    Dylan M. Gilley
#    dgilley@purdue.edu


import sys,argparse,time,random
import numpy as np
from copy import copy,deepcopy
from mol_classes import AtomList,IntraModeList
from lammps_files_classes import write_lammps_data,write_lammps_init
from parsers import parse_msf,parse_header


def main(argv):

    parser = argparse.ArgumentParser('Generate systems for running in hybridmdmc')

    # Positional arguments
    parser.add_argument(dest='system')

    parser.add_argument(dest='replicate')
    
    parser.add_argument(dest='molecule_types')

    parser.add_argument(dest='molecule_counts')

    # Optional arguments
    parser.add_argument('-msf', dest='msf', default='', type=str,
                        help='Name of the .msf file. If not specififed, script will default to prefix.msf')

    parser.add_argument('-header', dest='header', default='', type=str,
                        help='Name of the .header file. If not specififed, script will default to prefix.header')

    # Parse and store the inputs
    args = parser.parse_args()
    args.molecule_types = [_ for _ in args.molecule_types.split()]
    args.molecule_counts = [int(_) for _ in args.molecule_counts.split()]
    if not args.msf:
        args.msf = args.system + '.msf'
    if not args.header:
        args.header = args.system + '.header'
    run = args.system + '-' + args.replicate

    # Read in the .msf and .header files
    msf = parse_msf(args.msf)
    header = parse_header(args.header)
    atomtype2mass = {_[0]:_[1] for _ in header['masses']}

    # Determine the necessary box dimensions
    nodes_perside = int(np.ceil(np.sum([ args.molecule_counts[idx]*len(msf[_]['Atoms']) for idx,_ in enumerate(args.molecule_types) ])**(1./3.)))
    spacing = 6 # angstrom
    box_length = (nodes_perside+2)*spacing
    centers = np.array([[x,y,z]
                        for x in np.linspace(-box_length/2+spacing,box_length/2-spacing,num=nodes_perside)
                        for y in np.linspace(-box_length/2+spacing,box_length/2-spacing,num=nodes_perside)
                        for z in np.linspace(-box_length/2+spacing,box_length/2-spacing,num=nodes_perside)])
    
    # Place the molecules
    centers_order = list(range(len(centers)))
    random.shuffle(centers_order)
    molecules = [i for idx,_ in enumerate(args.molecule_types) for i in [_]*args.molecule_counts[idx] ]
    random.shuffle(molecules)
    atoms = {}
    interactions = {'Bonds':{},'Angles':{},'Dihedrals':{},'Impropers':{}}
    for idx,m in enumerate(molecules):

        # Center the molecule on the origin
        xyz = np.array([_[4:] for _ in msf[m]['Atoms']])
        xyz -= np.mean(xyz,axis=0)
        
        # Loop through all three dimensions, randomly rotating about each
        for dimension in range(3):
            theta = np.random.rand()*2*np.pi
            p1,p2 = np.array([0,0,0]),np.array([0,0,0])
            p1[dimension] = -1
            p2[dimension] = 1
            for pidx,point in enumerate(xyz):
                xyz[pidx] = PointRotate3D(p1,p2,point,theta)

        # Place the atoms and interactions into the appropriate dictionaries
        if 'Atoms' in msf[m].keys():
            tempmap = {}
            for aidx,atom in enumerate(msf[m]['Atoms']):
                tempmap[atom[0]] = len(atoms)+1
                atoms[tempmap[atom[0]]] = [idx] + atom[2:4] + (xyz[aidx] + centers[centers_order[idx]]).tolist()
        for inter in interactions.keys():
            if inter in msf[m].keys():
                for int_idx,int_ in enumerate(msf[m][inter]):
                    interactions[inter][len(interactions[inter])+1] = [int_[1]] + [tempmap[atom_] for atom_ in int_[2:]]

    # Create class instances for the atoms and interactions
    atomids = sorted(atoms.keys())
    atoms = AtomList(
        ids=atomids,
        lammps_type=[atoms[_][1] for _ in atomids],
        mol_id=[atoms[_][0] for _ in atomids],
        charge=[atoms[_][2] for _ in atomids],
        x=[atoms[_][3] for _ in atomids],
        y=[atoms[_][4] for _ in atomids],
        z=[atoms[_][5] for _ in atomids],
        mass=[atomtype2mass[atoms[_][1]] for _ in atomids]
    )
    interaction_instances = {
        inter_: IntraModeList(
            ids=sorted(interactions[inter_].keys()),
            lammps_type=[interactions[inter_][_][0] for _ in sorted(interactions[inter_].keys())],
            atom_ids=[interactions[inter_][_][1:] for _ in sorted(interactions[inter_].keys())],
        ) 
        for inter_ in interactions.keys()
    }

    # Write the LAMMPS data file
    write_lammps_data(
        run+'.in.data',
        atoms,
        interaction_instances['Bonds'],
        interaction_instances['Angles'],
        interaction_instances['Dihedrals'],
        interaction_instances['Impropers'],
        [[-box_length/2,box_length/2]]*3,
        charge=False,
        header=header
    )

    # Write the LAMMPS init file
    init = {
        'settings': '{}.in.settings'.format(args.system),
        'prefix': run,
        'data': run+'.in.data',
        'thermo_freq': 1000,
        'avg_freq': 1000,
        'dump2avg': 100,
        'coords_freq': 1000,
        'atom_style': 'molecular',
        'run_name':     [          'relax',           'density',              'heat',         'diffusion'],
        'run_type':     [      'nve/limit',               'npt',               'npt',               'npt'],
        'run_steps':    [             1000,              100000,            10000000,            10000000],
        'run_temp':     [[10.0,10.0,100.0], [100.0,100.0,100.0], [100.0,300.0,100.0], [300.0,300.0,100.0]],
        'run_press':    [  [1.0,1.0,100.0],     [1.0,1.0,100.0],     [1.0,1.0,100.0],     [1.0,1.0,100.0]],
        'run_timestep': [             0.25,                 1.0,                 1.0,                 1.0],
        'restart': False,
        'reset_steps': False,
        'thermo_keywords': ['temp','press','ke','pe'],
        'pair_style': 'lj/cut 9.0',
        'kspace_style': None,
        'bond_style': 'harmonic',
        'angle_Style': 'harmonic',
        'dihedral_style': 'opls',
        'improper_style': 'cvff'
    }
    write_lammps_init(init,run+'.in.init',step_restarts=False,final_restart=False,final_data=True)

    return


def PointRotate3D(p1, p2, p0, theta):
    from math import cos, sin, sqrt
    
    # Translate so axis is at origin
    p = p0 - p1

    # Initialize point q
    q = np.array([0.0,0.0,0.0])
    N = (p2-p1)
    Nm = sqrt(N[0]**2 + N[1]**2 + N[2]**2)

    # Rotation axis unit vector
    n = np.array([N[0]/Nm, N[1]/Nm, N[2]/Nm])
    
    # Matrix common factors
    c = cos(theta)
    t = (1 - cos(theta))
    s = sin(theta)
    X = n[0]
    Y = n[1]
    Z = n[2]

    # Matrix 'M'
    d11 = t*X**2 + c
    d12 = t*X*Y - s*Z
    d13 = t*X*Z + s*Y
    d21 = t*X*Y + s*Z
    d22 = t*Y**2 + c
    d23 = t*Y*Z - s*X
    d31 = t*X*Z - s*Y
    d32 = t*Y*Z + s*X
    d33 = t*Z**2 + c

    #            |p.x|
    # Matrix 'M'*|p.y|
    #            |p.z|
    q[0] = d11*p[0] + d12*p[1] + d13*p[2]
    q[1] = d21*p[0] + d22*p[1] + d23*p[2]
    q[2] = d31*p[0] + d32*p[1] + d33*p[2]

    # Translate axis and rotated point back to original location
    return q + p1

if __name__ == "__main__":
   main(sys.argv[1:])
