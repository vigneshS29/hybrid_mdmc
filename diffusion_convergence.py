#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

import os,sys,datetime
import numpy as np
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.parsers import *
from hybrid_mdmc.diffusion import *

# Main argument
def main(argv):
    """Driver for calculating diffusion convergence.
    """

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser()
    parser.add_argument(dest='timewindows', type=str)
    parser.HMDMC_parse_args()
    parser.adjust_default_args()
    args = parser.args
    args.timewindows = [int(_) for _ in args.timewindows.split()]

    # Read in the data_file, diffusion_file, rxndf, and msf files.
    atoms, bonds, angles, dihedrals, impropers, box, adj_mat, extra_prop = parse_data_file(
        args.data_file, unwrap=True, atom_style=args.atom_style)
    masterspecies = parse_msf(args.msf)
    rxndata = parse_rxndf(args.rxndf)

    # Initialize the output file
    with open(args.prefix+'.diffusionconvergence','w') as f:
        f.write(
            '# This file contains the average diffusion rates calculated using various sizes of time windows.\n'+\
            '# For each size of time window, the diffusion rates for each species for each voxel are caluclated\n'+\
            '# using the maximum number of non-overlapping time windows, beginning from the very first frame of\n'+\
            '# the trajectory file. The average and standard deviation of the resulting diffusion rates are then\n'+\
            '# calculated and printed here.\n#\n')
        f.write('# Trajectory file: {}\n'.format(args.trj_file))
        f.write('# Generation date: {}\n'.format(datetime.datetime.now()))
        f.write('{}\n\n'.format('-'*100))

    # Calculate the average diffusion rates
    reactivespecies = {k:v for k,v in masterspecies.items() if k in set([i for l in [_['reactant_molecules'] for _ in rxndata.values()] for i in l])}
    timesteps = get_trj_timesteps(args.trj_file)
    for window in args.timewindows:
        numberofwindows = int((timesteps[-1] - timesteps[0])/window)
        windowlength = int(len(timesteps)/numberofwindows)
        diffusion_rates = {}
        for windowidx in range(numberofwindows):
            diffusion_rates[windowidx] = calc_diffusionrate(
                args.trj_file,
                atoms,
                box,
                reactivespecies,
                args.num_voxels,
                xbounds=args.x_bounds,
                ybounds=args.y_bounds,
                zbounds=args.z_bounds,
                start=int(windowidx * windowlength),
                end=int((windowidx+1) * windowlength)
            )
        avg = {_:np.mean(np.ma.masked_invalid(np.array([diffusion_rates[idx][_] for idx in range(numberofwindows)])),axis=0) for _ in diffusion_rates[0].keys()}
        std = {_: np.std(np.ma.masked_invalid(np.array([diffusion_rates[idx][_] for idx in range(numberofwindows)])),axis=0) for _ in diffusion_rates[0].keys()}

        with open(args.prefix+'.diffusionconvergence','a') as f:
            f.write('window size {}\n'.format(window))
            for species in avg.keys():
                f.write('average {}\n'.format(species))
                for _ in avg[species]:
                    f.write('{}\n'.format(' '.join([str(__) for __ in _])))
                f.write('stddev {}\n'.format(species))
                for _ in std[species]:
                    f.write('{}\n'.format(' '.join([str(__) for __ in _])))

    return


def get_trj_timesteps(trjfile):
    timesteps = []
    flag = False
    with open(trjfile,'r') as f:
        for line in f:
            fields = line.split()
            if not fields: continue
            if fields[0] == 'ITEM:':
                if fields[1] == 'TIMESTEP':
                    flag = True
                    continue
            if flag:
                timesteps.append(int(float(fields[0])))
                flag = False
                continue
    return timesteps


if __name__ == '__main__':
    main(sys.argv[1:])