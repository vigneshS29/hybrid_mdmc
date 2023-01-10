#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu 

import os,argparse,sys,datetime,pipes,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import *
from copy import deepcopy
from hybridmdmc.classes import *
from hybridmdmc.parsers import *
from hybridmdmc.functions import *
from hybridmdmc.kmc import *
from mol_classes import AtomList,IntraModeList
from data_file_parser import parse_data_file
from lammps_files_classes import write_lammps_data,write_lammps_init


# Main argument.
def main(argv):    
    """Driver for conducting hybrid MD/MC simulation.

    Version 1.4
    -----------
    """

    # Create the argparse object
    parser = argparse.ArgumentParser(description='Conducts hybrid MD/MC simulation.')

    # Positional arguments
    parser.add_argument(dest='data_file', type=str,
                        help='Name of the LAMMPS data file to read in.')
    parser.add_argument(dest='diffusion_file', type=str,
                        help='Name of the diffusion file to read in.')

    # Optional arguments - files
    parser.add_argument('-prefix', dest='prefix', type=str, default='default',
                        help='Prefix of the rxndf file, msf file, and 
                        all output files. Prefix can be overridden for 
                        individual files. Default: data_file prefix')
    parser.add_argument('-conc_file', dest='conc_file', type=str, default='default',
                        help='Name of the concentration file. If not 
                        provided, the prefix is prepended to \.concentration"')
    parser.add_argument('-log_file', dest='log_file', type=str, default='default',
                        help='Name of the log file. If not provided, the
                        prefix is prepended to ".log"')
    parser.add_argument('-scale_file', dest='scale_file', type=str, default='default',
                        help='Name of the scale file. If not provided,
                        the prefix is prepended to ".log"')
    parser.add_argument('-settings', dest='settings', type=str, default='default',
                        help='Name of the settings file for LAMMPS MD
                        run. If not provided, the prefix is preprended
                        to ".in.settings"')
    parser.add_argument('-write_data', dest='write_data', type=str, default='default',
                        help='Name of the data file to write for LAMMPS
                        MD run. If not provided, the prefix is
                        preprended ".in.data"')
    parser.add_argument('-write_init', dest='write_init', type=str, default='default',
                        help='Name of the init file to write for LAMMPS
                        MD run. If not provided, the prefix is
                        preprended ".in.init"')
    parser.add_argument('-header', dest='header', type=str, default='default',
                        help='Name of the data header file. If not
                        provided, the rpefix is prepended to
                        ".header"')
    parser.add_argument('-rxndf', dest='rxndf', type=str, default='default',
                        help='Name of the rxndf file. If not provided,
                        the prefix is preprended to ".rxndf"')
    parser.add_argument('-msf', dest='msf', type=str, default='default',
                        help='Name of the msf file. If not provided,
                        the prefix is preprended to ".msf"')

    # Optional arguments - Diffusion information
    parser.add_argument('-temp', dest='temp', type=float, default=298.0,
                        help='Temperature of the simulation (K).
                        Default: 298.0 K')
    parser.add_argument('-relax', dest='relax', type=float, default=1.0e3,
                        help='Length of the relaxation nve/lim run.
                        Defualt: 1.0e3 (fs)')
    parser.add_argument('-diffusion', dest='diffusion', type=float, default=1.0e4,
                        help='Length of the diffusion npt run.
                        Defualt: 1.0e4 (fs)')

    # Optional arguments - KMC parameters
    parser.add_argument('-change_threshold', dest='change_threshold', type=float, default=0.10,
                        help='Fraction of original molecules to react
                        before this reactive KMC cycle completes.
                        Default: 0.10')
    parser.add_argument('-diffusion_cutoff', dest='diffusion_cutoff', type=float, default=0.4,
                        help='Minimum diffusion coefficient to consider
                        for possible reaction between voxels.
                        Default: 0.4')
    parser.add_argument('-kmc_type', dest='kmc_type', type=str, default='rejection_free',
                        help='Type of KMC to perform in each voxel.
                        Default: rejection_free')
    parser.add_argument('-diffusion_step', dest='diffusion_step', type=int, default=0,
                        help='Diffusion step. Defualt: 0.')
    parser.add_argument('-scale_prev_cycles', dest='scale_prev_cycles', type=float, default=30.0,
                        help='Number of MDMC cycles to consider when
                        performing rate scaling. Default: 30 cycles')
    parser.add_argument('-rxn_selection_threshold', dest='rxn_selection_threshold', type=int, default=0,
                        help='Number of times a reaction must have been
                        selected, per MDMC cycle, for its rate to be
                        scaled. Default: 5')
    parser.add_argument('-num_fraction_threshold', dest='num_fraction_threshold', type=float, default=0.01,
                        help='Maximum standard deviation of a species
                        number fraction for reaction rate scaling.
                        Default: 0.01')
    parser.add_argument('-max_scaling', dest='max_scaling', type=float, default=0.30,
                        help='Maxing scaling for reaction rates.
                        Default: 0.30')
    parser.add_argument('-scale_rates', dest='scale_rates', type=str, default='cumulative',
                        help='Option for scaling reaction rates when species mole fractions go stagnant.'+\
                        'Options: cumulative, static, off.
                        Default: cumulative')
    parser.add_argument('-cont_pause', dest='cont_pause', type=int, default=1,
                        help='')
    parser.add_argument('-progression_scale', dest='progression_scale', type=bool, default=False,
                        help='Option for using progression scaling.
                        Defualt: False')
    parser.add_argument('-rel_scale', dest='rel_scale', type=bool, default=True,
                        help='Option for using relative rate scaling.
                        Default: True')

    # Optional arguments - flags
    parser.add_argument('--debug', dest='debug', default=False, action='store_const', const=True)

    # Parse the command line arguments.
    args = parser.parse_args()
    args = adjust_arguments(args)
    if args == 'kill':
        quit()

    # Parse the data_file, diffusion_file, rxndf, and msf files
    atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(args.data_file,unwrap=True)
    voxels,diffusion_matrix = parse_diffusion(args.diffusion_file)
    rxn_data = parse_rxndf(args.rxndf)
    master_species = parse_msf(args.msf)
    data_header = parse_header(args.header)

    if args.debug:
        breakpoint()

    # Calculate the raw reaction rate for each reaction
    for r in rxn_data.keys():
        rxn_data[r]['rate'] = rxn_data[r]['A'][0]*args.temp**rxn_data[r]['b'][0]*np.exp(-rxn_data[r]['Ea'][0]/args.temp/0.00198588)

    if args.debug:
        breakpoint()

    # Create and populate an instance of the "MoleculeList" class
    voxelsmap,voxelsx,voxelsy,voxelsz = voxels2voxelsmap(voxels)
    voxelID2idx = {k:idx for idx,k in enumerate(sorted(list(voxels.keys())))}
    atomtypes2moltype = {}
    for k,v in master_species.items():
        atomtypes2moltype[tuple(sorted([i[2] for i in v['Atoms']]))] = k
    molecules = gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz)
    molID2molidx = {i:idx for idx,i in enumerate(molecules.ids)}

    # Check for consistency among the tracking files
    tfs = [ os.path.exists(f) for fidx,f in enumerate([args.conc_file,args.scale_file,args.log_file]) if [True,args.scale_rates,args.log][fidx] ]
    if len(set(tfs)) != 1:
        tf_names = [args.conc_file,args.scale_file,args.log_file]
        tf_requested = [True,args.scale_rates,args.log]
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
    for f in [(True,args.conc_file),(args.scale_rates,args.scale_file),(args.log,args.log_file)]:
        if f[0]:
            lines,new = [],False
            if not os.path.exists(f[1]) and '.concentration' in f[1]:
                lines = ['ReactionCycle  0\n',
                         'MoleculeCounts {}\n'.format(' '.join([ '{} {}'.format(sp,len([i for i in molecules.mol_types if i==sp])) for sp in sorted(master_species.keys()) ])),
                         'Time 0\n']
                new = True
            if not os.path.exists(f[1]) and '.scale' in f[1]:
                lines = ['ReactionCycle  0\n',
                         'ReactionTypes {}\n'.format(' '.join([ '{}'.format(i) for i in sorted(rxn_data.keys()) ])),
                         'ReactionScaling {}\n'.format(' '.join([ '1.0000' for i in sorted(rxn_data.keys()) ]))]
                new = True
            if not os.path.exists(f[1]) and '.log' in f[1]:
                new = True
            write_tracking_file(f[1],lines,driver=sys.argv[0],time=datetime.datetime.now(),step=args.step,new=new)

    if args.debug:
        breakpoint()

    # If rate scaling is requested, parse the concentration and scale files and create the progression object
    if args.scale_rates:
        counts,times,selected_rxns = parse_concentration(args.conc_file)
        progression,MDMCcycles_progression = get_progression(counts,times,selected_rxns,[0]+list(sorted(rxn_data.keys())),list(sorted(master_species.keys())),args.scale_prev_cycles)
        rxn_scaling,MDMCcycles_scaling = parse_scale(args.scale_file,args.scale_prev_cycles)
        # only the last row of the rxn_scaling is needed for this script
        rxn_scaling.drop(index=list(rxn_scaling.index)[:-1],inplace=True)

    # Consistency check
    if not np.all(sorted(list(rxn_data.keys()))==sorted(list(rxn_scaling.columns))):
        print('Error! The keys of the reaction data dictionary do not match the column headers fo the reaction scaling dataframe. Exiting...')
        if args.log:
            with open(args.logfile,'a') as f:
                f.write('rxn_data keys: {}\n'.format(sorted(list(rxn_data.keys()))))
                f.write('rxn_scaling columns: {}\n'.format(sorted(list(rxn_scaling.columns))))
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
    starting_mol_count,current_mol_count = len(sorted(set(atoms.mol_id))),len(sorted(set(atoms.mol_id)))
    add,delete,selected_rxn_types = {},[],[]
    Reacting,rxn_cycle = True,1
    while Reacting:
        
        if args.debug:
            breakpoint()

        # Loop over each voxel twice, once to get the total reaction rate for each, then again to carryout reactions
        Rtotal = {}
        vox_list = sorted(list(set(molecules.voxels)))
        for vox in vox_list:

            # Create and populate an instance of the "ReactionList" class, scaling if requested
            rxns = get_rxns(molecules,vox,voxelID2idx,diffusion_matrix,delete,rxn_data,args.diffusion_cutoff,args.temp)
            if args.scale_rates:
                rxns.rates = np.array([ r*rxn_scaling.loc[:,rxns.rxn_types[ridx]].iloc[0] for ridx,r in enumerate(rxns.rates) ])

            # Add the total rate for this voxel to the Rtotal dictionary
            Rtotal[vox] = np.sum(rxns.rates)

        Rmax = np.max(list(Rtotal.values()))

        if args.debug:
            breakpoint()

        for vox in vox_list:

            # Recalculate the rxns objects. This prioritizes memory over speed.
            # To prioritize speed over memory, save the rxns object for each voxel when calculated above.
            rxns = get_rxns(molecules,vox,voxelID2idx,diffusion_matrix,delete,rxn_data,args.diffusion_cutoff,args.temp)
            if args.scale_rates:
                rxns.rates = np.array([ r*rxn_scaling.loc[:,rxns.rxn_types[ridx]].iloc[0] for ridx,r in enumerate(rxns.rates) ])

            # Execute reaction(s)
            if len(rxns.ids) > 0:
                temp_add,temp_delete,dt,selected_rt = spkmc_rxn(rxns,rxn_data,molecules,Rmax,translate_distance=2.0)
                if add == 'kill':
                    print('Voxel: {} Rmax: {} Total rates: {}\nExiting...'.format(vox,Rmax,np.sum(rxns.rates)))
                    quit()
                add.update({ k+len(add):v for k,v in temp_add.items() })
                delete += [x for i in temp_delete for x in i]
            else:
                temp_delete = []
            current_mol_count -= len(temp_delete)
            selected_rxn_types += [selected_rt]

        # Check for completion
        if current_mol_count < (1-args.change_threshold)*starting_mol_count:
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

        if args.debug:
            breakpoint()

        # Update the topology
        atoms,bonds,angles,dihedrals,impropers = update_topology(master_species,molecules,add,delete,atoms,bonds,angles,dihedrals,impropers)

        # Create new molecules object
        molecules = gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz)
        molID2molidx = {i:idx for idx,i in enumerate(molecules.ids)}

        # Update the progression and rxn_scaling objects, if rate scaling is being performed
        if args.scale_rates:
            progression = update_progression(progression,molecules,list(sorted(master_species.keys())),dt,selected_rxn_types,args.scale_prev_cycles)

            if args.debug:
                breakpoint()

            rxn_scaling = scale_rates(rxn_scaling,progression,rxn_data,args.rxn_selection_threshold,args.num_fraction_threshold,args.max_scaling,style=args.scale_rates,cont_pause=args.cont_pause,progression_scale=args.progression_scale,rel_scale=args.rel_scale)

        if args.debug:
            breakpoint()

        # Update the tracking files
        lines = [ '\nReactionCycle {}\n'.format(rxn_cycle),
                  'MoleculeCounts {}\n'.format(' '.join([ '{} {}'.format(sp,len([i for i in molecules.mol_types if i==sp])) for sp in sorted(master_species.keys()) ])),
                  'SelectedReactionTypes {}\n'.format(' '.join([str(i) for i in selected_rxn_types])),
                  'Time {}\n'.format(dt)]
        with open(args.conc_file,'a') as f:
            for l in lines:
                f.write(l)

        lines = [ '\nReactionCycle {}\n'.format(rxn_cycle),
                  'ReactionTypes {}\n'.format(' '.join([ '{}'.format(i) for i in sorted(rxn_data.keys()) ])),
                  'ReactionScaling {}\n'.format(' '.join([ '{}'.format(rxn_scaling.loc[list(rxn_scaling.index)[-1],i]) for i in sorted(rxn_data.keys()) ]))]
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


if __name__ == '__main__':
    main(sys.argv[1:])
