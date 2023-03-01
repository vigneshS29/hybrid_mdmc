#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu 

import os,argparse,sys,datetime,pipes,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import *
from copy import deepcopy
from hybrid_mdmc.classes import *
from hybrid_mdmc.parsers import *
from hybrid_mdmc.functions import *
from hybrid_mdmc.kmc import *
from data_file_parser import parse_data_file
from lammps_files_classes import write_lammps_data,write_lammps_init

# Main argument
def main(argv):
    """Driver for conducting Hybrid MDMC simulation.
    """

    # Create the parser.
    parser = argparse.ArgumentParser(description='Conducts hybrid MD/MC simulation.')

    # Positional arguments
    parser.add_argument(dest='data_file', type=str,
                        help='Name of the LAMMPS data file to read in.')

    parser.add_argument(dest='diffusion_file', type=str,
                        help='Name of the diffusion file to read in.')

    # Optional arguments - files
    parser.add_argument('-prefix', dest='prefix', type=str, default='default',
                        help='Prefix of the rxndf file, msf file, and all output files. Prefix can be overridden for individual files. Default: data_file prefix')

    parser.add_argument('-conc_file', dest='conc_file', type=str, default='default',
                        help='Name of the concentration file. If not provided, the prefix is prepended to \.concentration"')

    parser.add_argument('-log_file', dest='log_file', type=str, default='default',
                        help='Name of the log file. If not provided, the prefix is prepended to ".log"')

    parser.add_argument('-scale_file', dest='scale_file', type=str, default='default',
                        help='Name of the scale file. If not provided, the prefix is prepended to ".log"')

    parser.add_argument('-settings', dest='settings', type=str, default='default',
                        help='Name of the settings file for LAMMPS MD run. If not provided, the prefix is preprended to ".in.settings"')

    parser.add_argument('-write_data', dest='write_data', type=str, default='default',
                        help='Name of the data file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.data"')

    parser.add_argument('-write_init', dest='write_init', type=str, default='default',
                        help='Name of the init file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.init"')

    parser.add_argument('-header', dest='header', type=str, default='default',
                        help='Name of the data header file. If not provided, the rpefix is prepended to ".header"')

    parser.add_argument('-rxndf', dest='rxndf', type=str, default='default',
                        help='Name of the rxndf file. If not provided, the prefix is preprended to ".rxndf"')

    parser.add_argument('-msf', dest='msf', type=str, default='default',
                        help='Name of the msf file. If not provided, the prefix is preprended to ".msf"')

    # Optional arguments - Diffusion information
    parser.add_argument('-temp', dest='temp', type=float, default=298.0,
                        help='Temperature of the simulation (K). Default: 298.0 K')

    parser.add_argument('-relax', dest='relax', type=float, default=1.0e3,
                        help='Length of the relaxation nve/lim run. Default: 1.0e3 (fs)')

    parser.add_argument('-diffusion', dest='diffusion', type=float, default=1.0e4,
                        help='Length of the diffusion npt run. Default: 1.0e4 (fs)')

    # Optional arguments - KMC parameters
    parser.add_argument('-change_threshold', dest='change_threshold', type=float, default=0.10,
                        help='Fraction of original molecules to react before this reactive KMC cycle completes.')

    parser.add_argument('-diffusion_cutoff', dest='diffusion_cutoff', type=float, default=0.4,
                        help='Minimum diffusion coefficient to consider for possible reaction between voxels.')

    parser.add_argument('-kmc_type', dest='kmc_type', type=str, default='rejection_free',
                        help='Type of KMC to perform in each voxel.')

    parser.add_argument('-diffusion_step', dest='diffusion_step', type=int, default=0,
                        help='Diffusion step. Default: 0.')

    parser.add_argument('-scalerates', dest='scalerates', type=str, default='cumulative',
                        help='Option for scaling reaction rates when species mole fractions go stagnant. Options: cumulative, static, off.')

    parser.add_argument('-scalingcriteria_rollingmean_stddev', dest='scalingcriteria_rollingmean_stddev', type=float, default=0.1,
                        help='Maximum (less than or equal to) standard deviation of the rolling mean of the number fraction of a molecular species'+\
                        'for that number fraction to be considered stagnant.')

    parser.add_argument('-scalingcriteria_rollingmean_cycles', dest='scalingcriteria_rollingmean_cycles', type=int, default=3,
                        help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')

    parser.add_argument('-scalingcriteria_concentration_slope', dest='scalingcriteria_concentration_slope', type=float, default=0.1,
                        help='Maximum (less than or equal to) slope of the number fraction of a molecular species for that number fraction to be considered stagnant.')

    parser.add_argument('-scalingcriteria_concentration_cycles', dest='scalingcriteria_concentration_cycles', type=int, default=3,
                        help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')

    parser.add_argument('-windowsize_rollingmean', dest='windowsize_rollingmean', type=int, default=3,
                        help='Window size for the calculation of the rolling mean, measured in MDMC cycles.')

    parser.add_argument('-windowsize_slope', dest='windowsize_slope', type=int, default=5,
                        help='Window size for the calculation of the slope of concentration, measured in MDMC cycles.')

    parser.add_argument('-windowsize_scalingpause', dest='windowsize_scalingpause', type=int, default=3,
                        help='Number of MDMC cycles after a reaciton is scaled or unscaled before it can be scaled again.')

    parser.add_argument('-scalingfactor_adjuster', dest='scalingfactor_adjuster', type=float, default=0.1,
                        help='Quantity which a reaction rate is multiplied by to scale that rate.')

    parser.add_argument('-scalingfactor_minimum', dest='scalingfactor_minimum', type=float, default=1e-6,
                        help='Minimum reaction rate scaling factor for all reactions.')

    # Optional arguments - flags
    parser.add_argument('--debug', dest='debug', default=False, action='store_const', const=True)

    parser.add_argument('--log', dest='log', default=False, action='store_const', const=True)

    # Parse the command line arguments.
    args = parser.parse_args()
    args = adjust_arguments(args)

    # Parse the data_file, diffusion_file, rxndf, and msf files
    atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(args.data_file,unwrap=True)
    voxels,diffusion_matrix = parse_diffusion(args.diffusion_file)
    rxndata = parse_rxndf(args.rxndf)
    masterspecies = parse_msf(args.msf)
    data_header = parse_header(args.header)
    rxnmatrix = get_rxnmatrix(rxndata,masterspecies)

    if args.debug:
        breakpoint()

    # Calculate the raw reaction rate for each reaction
    for r in rxndata.keys():
        rxndata[r]['rawrate'] = rxndata[r]['A'][0]*args.temp**rxndata[r]['b'][0]*np.exp(-rxndata[r]['Ea'][0]/args.temp/0.00198588)

    if args.debug:
        breakpoint()

    # Create and populate an instance of the "MoleculeList" class
    voxelsmap,voxelsx,voxelsy,voxelsz = voxels2voxelsmap(voxels)
    voxelID2idx = {k:idx for idx,k in enumerate(sorted(list(voxels.keys())))}
    atomtypes2moltype = {}
    for k,v in masterspecies.items():
        atomtypes2moltype[tuple(sorted([i[2] for i in v['Atoms']]))] = k
    molecules = gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz)
    molID2molidx = {i:idx for idx,i in enumerate(molecules.ids)}

    # Check for consistency among the tracking files
    tfs = [ os.path.exists(f) for fidx,f in enumerate([args.conc_file,args.scale_file,args.log_file]) if [True,args.scalerates,args.log][fidx] ]
    if len(set(tfs)) != 1:
        tf_names = [args.conc_file,args.scale_file,args.log_file]
        tf_requested = [True,args.scalerates,args.log]
        print('Error! Inconsistent exsitence of tracking files.')
        print('  File Name                Requested   Exists')
        for idx in range(len(tf_names)):
            r,e = 'True','True'
            if not tf_requested[idx]: r = 'False'
            if not os.path.exists(tf_names[idx]): e = 'False'
            print('  {:<24s} {:<9s}   {:<6s}'.format(tf_names[idx],r,e))
        print('Exiting...')
        quit()

    # Create/append the tracking files
    for f in [(True,args.conc_file),(args.scalerates,args.scale_file),(args.log,args.log_file)]:
        if f[0]:
            lines,new = [],False
            if not os.path.exists(f[1]) and '.concentration' in f[1]:
                lines = ['ReactionCycle  0\n',
                         'MoleculeCounts {}\n'.format(' '.join([ '{} {}'.format(sp,len([i for i in molecules.mol_types if i==sp])) for sp in sorted(masterspecies.keys()) ])),
                         'Time 0\n']
                new = True
            if not os.path.exists(f[1]) and '.scale' in f[1]:
                lines = ['ReactionCycle  0\n',
                         'ReactionTypes {}\n'.format(' '.join([ '{}'.format(i) for i in sorted(rxndata.keys()) ])),
                         'ReactionScaling {}\n'.format(' '.join([ '1.0000' for i in sorted(rxndata.keys()) ]))]
                new = True
            if not os.path.exists(f[1]) and '.log' in f[1]:
                new = True
            write_tracking_file(f[1],lines,driver=sys.argv[0],time=datetime.datetime.now(),step=args.diffusion_step,new=new)

    if args.debug:
        breakpoint()

    # If rate scaling is requested, parse the concentration and scale files and create the progression object
    if args.scalerates:
        counts,times,selected_rxns = parse_concentration(args.conc_file)
        if counts == 'kill':
            quit()
        progression,MDMCcycles_progression = get_progression(counts,times,selected_rxns,[0]+list(sorted(rxndata.keys())),list(sorted(masterspecies.keys())))
        rxnscaling,MDMCcycles_scaling = parse_scale(args.scale_file)

    # Consistency check
    if not np.all(sorted(list(rxndata.keys())) == sorted(list(rxnscaling.columns))):
        print('Error! The keys of the reaction data dictionary do not match the column headers of the reaction scaling dataframe. Exiting...')
        if args.log:
            with open(args.logfile,'a') as f:
                f.write('rxndata keys: {}\n'.format(sorted(list(rxndata.keys()))))
                f.write('rxnscaling columns: {}\n'.format(sorted(list(rxnscaling.columns))))
        quit()
    if not np.all( sorted(MDMCcycles_progression) == sorted(MDMCcycles_scaling) ):
        print('Error! MDMC cycles generated from \"get_progression\" and \"parse_scale\" are inconsistent. Exiting...')
        if args.log:
            with open(args.logfile,'a') as f:
                f.write('MDMCcycles_progression {}\n'.format(sorted(MDMCcycles_progression)))
                f.write('MDMCcycles_scaling: {}\n'.format(sorted(list(MDMCcycles_scaling))))
        quit()

    if args.debug:
        breakpoint()

    # Begin the KMC loop
    molecount_starting,molecount_current = len(sorted(set(atoms.mol_id))),len(sorted(set(atoms.mol_id)))
    add,delete,selected_rxn_types = {},[],[]
    Reacting,rxn_cycle = True,1
    while Reacting:
        
        if args.debug:
            breakpoint()

        # Loop over each voxel to get the reactions available in each voxel.
        vox_list = sorted(list(set(molecules.voxels)))
        rxns_byvoxel = {vox:[] for vox in vox_list}
        for vox in vox_list:

            # Create and populate an instance of the "ReactionList" class.
            rxns = get_rxns(molecules,vox,voxelID2idx,diffusion_matrix,delete,rxndata,args.diffusion_cutoff,args.temp)

            # Add the reaction data for this voxel to rxns_byvoxel.
            rxns_byvoxel[vox] = [(rxns.rxn_types[idx],_) for idx,_ in enumerate(rxns.rates)]

        if args.debug:
            breakpoint()

        # If scaling is requested, unscale and scale the reactions
        if args.scalerates:
            rxnscaling = ratescaling_unscalerxns(rxnmatrix,rxnscaling,progression,cycle=progression.index[-1])
            rxnscaling = ratescaling_scalerxns(
                rxndata,rxnmatrix,rxnscaling,progression,
                args.windowsize_slope,args.windowsize_scalingpause,
                args.scalingcriteria_concentration_slope,args.scalingcriteria_concentration_cycles,
                args.scalingfactor_adjuster,args.scalingfactor_minimum,
                rxnlist=sorted(list(set([ _[0] for vox in vox_list for _ in rxns_byvoxel[vox] ]))))

        # Calculate the maximum voxel total reaction rate.
        Rmax = np.max([ np.sum([rxnscaling.loc[rxnscaling.index[-1],_[0]]*_[1] for _ in rxns_byvoxel[vox]]) for vox in vox_list ])

        # pdb breakpoint
        if args.debug:
            breakpoint()

        # Loop over each voxel again. To prioritize memory over speed,
        # reactions are rediscoverd for each voxel. Then, each voxel
        # undergoes KMC selection of a reaction.
        for vox in vox_list:

            # Recalculate the rxns objects. This prioritizes memory over speed.
            # To prioritize speed over memory, save the rxns object for each voxel when calculated above.
            rxns = get_rxns(molecules,vox,voxelID2idx,diffusion_matrix,delete,rxndata,args.diffusion_cutoff,args.temp)
            if args.scalerates:
                rxns.rates = np.array([ r*rxnscaling.loc[rxnscaling.index[-1],rxns.rxn_types[ridx]] for ridx,r in enumerate(rxns.rates) ])

            # Execute reaction(s)
            temp_delete = []
            if len(rxns.ids) > 0:
                temp_add,temp_delete,dt,selected_rt = spkmc_rxn(rxns,rxndata,molecules,Rmax,translate_distance=2.0)
                if temp_add == 'kill':
                    print('Voxel: {} Rmax: {} Total rates: {}\nExiting...'.format(vox,Rmax,np.sum(rxns.rates)))
                    quit()
                add.update({ k+len(add):v for k,v in temp_add.items() })
                delete += [x for i in temp_delete for x in i]
            molecount_current -= len(temp_delete)
            selected_rxn_types += [selected_rt]

        # Check for completion
        if molecount_current < (1-args.change_threshold)*molecount_starting:
            Reacting = False

        # Check for erroneous double deletion
        if len(set(delete)) != len(delete):
            print('Error! Molecules deleted twice. Exiting...')
            quit()

        # If requested, print debugging information to the log file
        if args.log:
            with open(args.log_file,'a') as f:
                f.write('add: {}\n'.format(add))
                f.write('delete: {}\n'.format(delete))
                f.write('selected_rxn_types: {}\n\n'.format(selected_rxn_types))

        # pdb breakpoint
        if args.debug:
            breakpoint()

        # Update the topology
        atoms,bonds,angles,dihedrals,impropers = update_topology(masterspecies,molecules,add,delete,atoms,bonds,angles,dihedrals,impropers)

        # Create new molecules object
        molecules = gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz)
        molID2molidx = {i:idx for idx,i in enumerate(molecules.ids)}

        # Update the progression objects, if rate scaling is being performed
        if args.scalerates:
            progression = update_progression(progression,molecules,list(sorted(masterspecies.keys())),dt,selected_rxn_types)

        # pdb breakpoint
        if args.debug:
            breakpoint()

        # Update the tracking files
        lines = [ '\nReactionCycle {}\n'.format(rxn_cycle),
                  'MoleculeCounts {}\n'.format(' '.join([ '{} {}'.format(sp,len([i for i in molecules.mol_types if i==sp])) for sp in sorted(masterspecies.keys()) ])),
                  'SelectedReactionTypes {}\n'.format(' '.join([str(i) for i in selected_rxn_types])),
                  'Time {}\n'.format(dt)]
        with open(args.conc_file,'a') as f:
            for l in lines:
                f.write(l)

        lines = [ '\nReactionCycle {}\n'.format(rxn_cycle),
                  'ReactionTypes {}\n'.format(' '.join([ '{}'.format(i) for i in sorted(rxndata.keys()) ])),
                  'ReactionScaling {}\n'.format(' '.join([ '{}'.format(rxnscaling.loc[list(rxnscaling.index)[-1],i]) for i in sorted(rxndata.keys()) ]))]
        with open(args.scale_file,'a') as f:
            for l in lines:
                f.write(l)

        # Update the tracking
        rxn_cycle += 1
        add,delete,selected_rxn_types = {},[],[]

    # Write the resulting data and init files
    init = {
        'settings': args.settings,
        'prefix': args.prefix,
        'data': args.write_data,
        'thermo_freq': 1000,
        'avg_freq': 1000,
        'dump4avg': 100,
        'coords_freq': 1000,
        'run_name': [ 'relax','diffusion' ],
        'run_type': [ 'nve/limit','npt' ],
        'run_steps': [ args.relax, args.diffusion],
        'run_temp': [ [args.temp,args.temp,10.0],[args.temp,args.temp,100.0] ],
        'run_press': [ [1.0,1.0,100.0],[1.0,1.0,100.0] ],
        'run_timestep': [0.25,0.25],
        'restart': False,
        'reset_steps': False,
        'thermo_keywords': ['temp', 'press', 'ke', 'pe']}
    init['pair_style'] = 'lj/cut 9.0'
    init['kspace_style'] = None
    init['angle_style'] = 'harmonic'
    init['dihedral_style'] = 'opls'
    init['improper_style'] = 'cvff'
    init['neigh_modify'] = 'every 1 delay 0 check yes one 10000'
    init['coords_freq'] = 20
    write_lammps_data(args.write_data,atoms,bonds,angles,dihedrals,impropers,box,header=data_header)
    write_lammps_init(init,args.write_init,step_restarts=False,final_restart=False,final_data=True)

    return


def adjust_arguments(args):

    if args.prefix == 'default':
        args.prefix = args.data_file
        if args.data_file.endswith('.in.data'):
            args.prefix = args.data_file[:-8]
        elif args.data_file.endswith('.end.data'):
            args.prefix = args.data_file[:-9]
        elif args.data_file.endswith('.data'):                                                                                                                                                      
            args.prefix = args.data_file[:-5]
    if args.rxndf == 'default':
        args.rxndf = args.prefix+'.rxndf'
    if args.msf == 'default':
        args.msf = args.prefix+'.msf'
    if args.settings == 'default':
        args.settings = args.prefix+'.in.settings'
    if args.header == 'default':
        args.header = args.prefix+'.header'
    if args.write_data == 'default':
        args.write_data = args.prefix+'.in.data'
    if args.write_init == 'default':
        args.write_init = args.prefix+'.in.init'
    if args.conc_file == 'default':
        args.conc_file = args.prefix+'.concentration'
    if args.log_file == 'default':
        args.log_file = args.prefix+'.log'
    if args.scale_file == 'default':
        args.scale_file = args.prefix+'.scale'

    return args


if __name__ == '__main__':
    main(sys.argv[1:])
