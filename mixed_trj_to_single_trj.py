#!/usr/bin/env python3
# Author
#    Dylan M. Gilley
#    dgilley@purdue.edu

import os,argparse,sys,datetime
import numpy as np
import pandas as pd
from copy import deepcopy
from mol_classes import AtomList,IntraModeList
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.frame_generator import frame_generator
from hybrid_mdmc.parsers import parse_msf

# Main argument
def main(argv):
    """
    """

    # Create the parser object.
    parser = argparse.ArgumentParser()

    # Positional arguments.
    parser.add_argument(dest='trj_file', type=str,
                        help='Name of the trajectory file.')

    # Optional arguments.
    parser.add_argument('-lammps_data_file', dest='lammps_data_file', type=str, default='default',
                        help='Name of the LAMMPS data file to read in. If not specified, the trajectory file prefix is prepended to .end.data')

    parser.add_argument('-msf_file', dest='msf_file', type=str, default='default',
                        help='Name of the master species file. If not specified, the trajectory file prefix is prepended to .msf')

    parser.add_argument('-start', dest='start', type=int, default=0)

    parser.add_argument('-end', dest='end', type=int, default=-1)

    parser.add_argument('-every', dest='every', type=int, default=1)

    # Parse the input arguments.
    args = parser.parse_args()
    prefix = args.trj_file[:-10]
    if args.lammps_data_file == 'default':
        args.lammps_data_file = prefix + '.end.data'
    if args.msf_file == 'default':
        args.msf_file = prefix + '.msf'

    # Read in the LAMMPS data file and master species file (.msf).
    atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(args.lammps_data_file)
    species_data = parse_msf(args.msf_file)

    # Create one dictionary to map atom ID to molecule type,
    # and another to map molecule type to all atom IDs in molecules of that type.
    atomID2moleculetype = gen_atomID2moleculetype(atoms,species_data)
    species2atomIDs = {
        species:sorted([
            _ for _ in atoms.ids if atomID2moleculetype[_] == species
        ])
        for species in species_data.keys()
    }

    # Initialize a trj file for each molecule type.
    for species in species_data.keys():
        with open(prefix+'_{}.lammpstrj'.format(species),'w') as f:
            f.write('')

    # Loop through the trj file using frame_generator.
    for atoms,timestep,box,prop in frame_generator(args.trj_file,start=args.start,end=args.end,every=args.every,unwrap=False,return_prop=True):
        for species in species_data.keys():
            #temp_atoms = deepcopy(atoms)
            #deletion_idxs = temp_atoms.get_idx(ids=[_ for _ in temp_atoms.ids if _ not in species2atomIDs[species]])
            #temp_atoms.del_idx(idx=deletion_idxs,reassign_ids=False,reassign_lammps_type=False)
            indices = sorted([index for index,_ in enumerate(atoms.ids) if _ in species2atomIDs[species]])
            with open(prefix+'_{}.lammpstrj'.format(species),'a') as f:
                f.write('ITEM: TIMESTEP\n{}\n'.format(timestep))
                f.write('ITEM: NUMBER OF ATOMS\n{}\n'.format(len(indices)))
                f.write('ITEM: BOX BOUNDS pp pp pp\n{} {}\n{} {}\n{} {}\n'.format(box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1]))
                f.write('ITEM: ATOMS {}\n'.format(' '.join(sorted(list(prop.keys())))))
                for idx in indices:
                    f.write('{}\n'.format(' '.join([str(prop[_][idx]) for _ in sorted(list(prop.keys()))])))
    
    return


def gen_atomID2moleculetype(atoms,species_data):

    atomtypes2moleculetype = { tuple(sorted([_[2] for _ in v['Atoms']])):k for k,v in species_data.items() }
    atomID2moleculetype = {
        atomID:atomtypes2moleculetype[
            tuple(sorted([
                _ for molidx,_ in enumerate(atoms.lammps_type) if atoms.mol_id[molidx] == atoms.mol_id[atomidx]
            ]))
        ]
        for atomidx,atomID in enumerate(atoms.ids)
    }

    return atomID2moleculetype

if __name__ == '__main__':
    main(sys.argv[1:])
