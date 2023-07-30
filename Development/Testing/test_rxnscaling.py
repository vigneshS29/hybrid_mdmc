#!/usr/bin/env python3
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

import numpy as np
import pandas as pd
from hybrid_mdmc.classes import *
from hybrid_mdmc.parsers import *
from hybrid_mdmc.functions import *
from hybrid_mdmc.data_file_parser import parse_data_file

# Import .in.data, .diffusion, .rxndf, and .msf files.
atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file('test_rxnscaling.in.data',unwrap=True)
voxels,diffusion_matrix = parse_diffusion('test_rxnscaling.diffusion')
rxndata = parse_rxndf('test_rxnscaling.rxndf')
masterspecies = parse_msf('test_rxnscaling.msf')
rxnmatrix = get_rxnmatrix(rxndata,masterspecies)
breakpoint()

# Create and populate an instance of the "MoleculeList" class.
voxelsmap,voxelsx,voxelsy,voxelsz = voxels2voxelsmap(voxels)
voxelID2idx = {k:idx for idx,k in enumerate(sorted(list(voxels.keys())))}
atomtypes2moltype = {}
for k,v in masterspecies.items():
    atomtypes2moltype[tuple(sorted([i[2] for i in v['Atoms']]))] = k
molecules = gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz)
molID2molidx = {i:idx for idx,i in enumerate(molecules.ids)}
breakpoint()

# Import .scale and .concentration files.
counts,times,selected_rxns = parse_concentration('test_rxnscaling.concentration')
if counts == 'kill':
    quit()
progression,MDMCcycles_progression = get_progression(counts,times,selected_rxns,[0]+list(sorted(rxndata.keys())),list(sorted(masterspecies.keys())))
rxnscaling,MDMCcycles_scaling = parse_scale('test_rxnscaling.scale')
breakpoint()

# Attempt reaction scaling
windowsize_slope = 20
windowsize_scalingpause = 1
windowsize_rxnselection = 20
scalingcriteria_concentration_slope = 100
scalingcriteria_concentration_cycles = 1
scalingcriteria_rxnselection_count = 0
scalingfactor_adjuster = 0.1
scalingfactor_minimum = 1e-10
breakpoint()

vox_list = sorted(list(set(molecules.voxels)))
PSSrxns = get_PSSrxns(
    rxndata,rxnmatrix,rxnscaling,progression,
    windowsize_slope,windowsize_scalingpause,
    scalingcriteria_concentration_slope,scalingcriteria_concentration_cycles)
rxnscaling = ratescaling_unscalerxns(
    rxnmatrix,rxnscaling,progression,PSSrxns,cycle=progression.index[-1])
rxnscaling = ratescaling_scalerxns(
    rxndata,rxnmatrix,rxnscaling,progression,PSSrxns,
    scalingcriteria_rxnselection_count,
    windowsize_scalingpause,windowsize_rxnselection,
    scalingfactor_adjuster,scalingfactor_minimum,
    rxnlist=sorted(list(set([ _[0] for vox in vox_list for _ in rxns_byvoxel[vox] ]))))
breakpoint()

for vox in vox_list:
    rxns = get_rxns(molecules,vox,voxelID2idx,diffusion_matrix,delete,rxndata,args.diffusion_cutoff,args.temp)
    rxns_byvoxel[vox] = [(rxns.rxn_types[idx],_) for idx,_ in enumerate(rxns.rates)]
breakpoint()
