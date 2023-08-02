#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu 


import numpy as np
import pandas as pd
from scipy.spatial.distance import *
from copy import deepcopy
from sklearn import linear_model
from hybrid_mdmc.classes import MoleculeList,ReactionList


def voxels2voxelsmap(voxels):
    """Creates an object to map x/y/z indices to a voxel

    Parameters
    ----------
    voxels: dictionary
        - keys: int
            > voxel ID
        - values: list of lists
            > each sublist is [lmin,lmax]

    Returns
    -------
    voxelmap: dictionary
        > keys: tuple
            - (xidx,yidx,zidx): indices of minimum voxel bounds,
              indexed to x/y/z arrays
        > values: list
            - [ voxel ID, [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    x: np.array
        holds the left-most bounds for voxels, in the x direction
    y: np.array
        holds the left-most bounds for voxels, in the y direction
    z: np.array
        holds the left-most bounds for voxels, in the z direction
    """

    voxelsb = { tuple([tuple(i) for i in v]):k for k,v in voxels.items() }
    x = sorted(set([ i for bounds in [v[0] for v in voxels.values()] for i in bounds  ]))
    y = sorted(set([ i for bounds in [v[1] for v in voxels.values()] for i in bounds  ]))
    z = sorted(set([ i for bounds in [v[2] for v in voxels.values()] for i in bounds  ]))
    voxelsmap = {}
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            for k in range(len(z)-1):
                xx = [x[i],x[i+1]]
                yy = [y[j],y[j+1]]
                zz = [z[k],z[k+1]]
                voxelsmap[(i,j,k)] = [ voxelsb[(tuple(xx),tuple(yy),tuple(zz))],xx,yy,zz]
                if i == len(x)-2:
                    voxelsmap[(i+1,j,k)] = [ voxelsb[(tuple(xx),tuple(yy),tuple(zz))],xx,yy,zz]
                if j == len(y)-2:
                    voxelsmap[(i,j+1,k)] = [ voxelsb[(tuple(xx),tuple(yy),tuple(zz))],xx,yy,zz]
                if k == len(z)-2:
                    voxelsmap[(i,j,k+1)] = [ voxelsb[(tuple(xx),tuple(yy),tuple(zz))],xx,yy,zz]

    return voxelsmap,np.array(x),np.array(y),np.array(z)


def gen_molecules(atoms,atomtypes2moltype,voxelsmap,voxelsx,voxelsy,voxelsz):
    """Creates an instance of the MoleculeList class

    Parameters
    ----------
    atoms: instance of AtomList class
    atomtypes2moltype: dict
        > keys: tuple
            - sorted tuple of atom types in the molecule
        > values: str
            - species name
    voxelmap: dict
        > keys: tuple
            - (xidx,yidx,zidx): indices of minimum voxel bounds,
               indexed to x/y/z arrays
        > values: list
            - [ voxel ID, [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    voxelsx: np.array
        holds the left-most bounds for voxels, in the x direction
    voxelsy: np.array
        holds the left-most bounds for voxels, in the y direction
    voxelsz: np.array
        holds the left-most bounds for voxels, in the z direction

    Returns
    -------
    MoleculeList: instance of MoleculeList
    """

    voxelbounds = [voxelsx,voxelsy,voxelsz]
    molecules_dict = {m:{} for m in set(atoms.mol_id)}
    for m in molecules_dict.keys():
        idxs = [ idx for idx in range(len(atoms.ids)) if atoms.mol_id[idx]==m ]
        molecules_dict[m]['atom_IDs'] = sorted(atoms.ids[idxs].tolist())
        molecules_dict[m]['cog'] = np.mean(np.array([atoms.x[idxs],atoms.y[idxs],atoms.z[idxs]]),axis=1)
        cog = molecules_dict[m]['cog']
        # For assigning voxels, wrap the COG back into the simulation box
        box = [ [voxelsx[0],voxelsx[-1]], [voxelsy[0],voxelsy[-1]], [voxelsz[0],voxelsz[-1]] ]
        for i in range(3):
            while cog[i] < box[i][0]:
                cog[i] += box[i][1]-box[i][0]
            while cog[i] > box[i][1]:
                cog[i] -= box[i][1]-box[i][0]
        molecules_dict[m]['type'] = atomtypes2moltype[tuple(sorted(atoms.lammps_type[idxs]))]
        molecules_dict[m]['voxel'] = voxelsmap[(
            np.argwhere(voxelsx<=cog[0])[-1][0], 
            np.argwhere(voxelsy<=cog[1])[-1][0],
            np.argwhere(voxelsz<=cog[2])[-1][0] )][0]
    mkeys = sorted(list(molecules_dict.keys()))
    return MoleculeList(
        ids=mkeys,
        atom_ids=[molecules_dict[m]['atom_IDs'] for m in mkeys],
        cogs=[molecules_dict[m]['cog'] for m in mkeys],
        mol_types=[molecules_dict[m]['type'] for m in mkeys],
        voxels=[molecules_dict[m]['voxel'] for m in mkeys])


def get_rxns(molecules,vox,voxelID2idx,diffusion_matrix,delete,rxn_data,diffusion_cutoff,temp):
    """Creates instance of class hybridmdmc.classes.ReactionList.

    Parameters
    ----------
    molecules: instance of class HybridMDMC_Classes.MoleculeList
    vox: int
        Voxel number.
    voxelID2idx: dict
        Maps voxel IDs to their index in the diffusion matrix.
    diffusion_matrix: np.ndarray
        NxN array, where N is the number of voxels.
        An entry represents the diffusion coefficient going from
        voxel number (row index - 1) to voxel number (column index - 1).
    delete: list of int
        Molecules IDs that are scheduled for deletion.
    rxn_data: dict
        keys: int
            Reaction number.
        values: dict
            'reactant_molecules': list
                A list of the molecule type(s) that participate in the
                given reaction. Format: [molecule_type, ...]
            'product_molecules': list
                A list of the molecule type(s) that result from the
                given reaction. Format: [molecule_type, ...]
            'A': [float]
                A in k=A(T^b)exp(-Ea/RT).
            'b': [float]
                b in k=A(T^b)exp(-Ea/RT).
            'Ea': [float]
                Ea in k=A(T^b)exp(-Ea/RT).
    diffusion_cutoff: float
        Cutoff distance for considering diffusion.
    temp: float
        Temperature.

    Returns
    -------
    rxns: instance of class HybridMDMC_Classes.ReactionList
    """

    molID2idx = {ID:idx for idx,ID in enumerate(molecules.ids)} # Create dictionary to map molecule IDs to their index in the molecules object
    rmol_IDs = sorted([ molecules.ids[idx] for idx in range(len(molecules.ids)) if molecules.voxels[idx] == vox ]) # Get molecule IDs for the reactive species
    rxns,rxn_count = {},1

    for rxn_type in rxn_data.keys(): # Loop over every possible reaction in the rxn_data
        for rmID in rmol_IDs: # Loop over each reactive molecule in the voxel
            #if rmID in delete: continue # If this molecule has already reacted, skip it
            if molecules.mol_types[molID2idx[rmID]] not in rxn_data[rxn_type]['reactant_molecules']: continue # Skip if this molecule cannot undergo this rxn
            reactant_types = deepcopy(rxn_data[rxn_type]['reactant_molecules'])
            reactant_types.remove(molecules.mol_types[molID2idx[rmID]]) # List of remaining reactant molecule types, if any, after removing the active molecule type

            # Unimolecular rxn
            if len(reactant_types) == 0:
                rxns[rxn_count] = {
                    'rxn_type': rxn_type,
                    'reactants': [rmID],
                    'rate': rxn_data[rxn_type]['A'][0]*temp**rxn_data[rxn_type]['b'][0]*np.exp(-rxn_data[rxn_type]['Ea'][0]/temp/0.00198588)}
                rxn_count += 1
                continue

            # Bimolecular rxn
            if len(reactant_types) == 1:
                partner_IDs = [ID for ID in molecules.ids
                               if diffusion_matrix[voxelID2idx[vox],voxelID2idx[molecules.voxels[molID2idx[ID]]]] >= diffusion_cutoff
                               and molecules.mol_types[molID2idx[ID]] == reactant_types[0]
                               #and ID not in delete
                               and ID != rmID]
                for pID in partner_IDs:
                    rxns[rxn_count] = {
                        'rxn_type': rxn_type,
                        'reactants': [rmID,pID],
                        'rate': rxn_data[rxn_type]['A'][0]*temp**rxn_data[rxn_type]['b'][0]*np.exp(-rxn_data[rxn_type]['Ea'][0]/temp/0.00198588)
                        *diffusion_matrix[voxelID2idx[vox],voxelID2idx[molecules.voxels[molID2idx[pID]]]]}
                    rxn_count += 1
                continue

            # More than bimolecular rxn
            if len(reactant_types) > 1:
                print('Error! hybridmdmc.functions.get_rxns does not currently support reactions between more than 2 molecules. Crashed in voxel {}, reaction number {}.'.format(vox,rxn_type))

    rxns = ReactionList(
        ids=sorted(list(rxns.keys())),
        rxn_types=[rxns[k]['rxn_type'] for k in sorted(list(rxns.keys())) ],
        reactants=[rxns[k]['reactants'] for k in sorted(list(rxns.keys())) ],
        rates=[rxns[k]['rate'] for k in sorted(list(rxns.keys())) ])

    return rxns


def update_topology(msf,molecules,add,delete,atoms,bonds,angles,dihedrals,impropers):
    """Updates the topology a given system.

    Parameters
    ----------
    msf: dict
        keys: str
            Species name.
        values: dict
            'Atoms': list of lists
                [[atom ID, mol ID, atom type, q, x, y, z], ...]
            'Bonds'/'Angles'/'Dihedrals'/'Impropers': list of lists
                [[interaction ID, interaction type, atom i ID, atom j ID, ...] ...]
    molecules: instance of class HybridMDMC_Classes.MoleculeList
    add: dict
        keys: int
            New molecule ID.
        values: list
            [ species type (str), cog (np.ndarray) ]
    delete: list of int
        Molecule IDs scheduled for deletion.
    atoms: instance of class mol_classes.AtomList
    bonds: instance of class mol_classes.IntraModeList
    angles: instance of class mol_classes.IntraModeList
    dihedrals: instance of class mol_classes.IntraModeList
    impropers: instance of class mol_classes.IntraModeList

    Returns
    -------
    atoms: instance of class mol_classes.AtomList
    bonds: instance of class mol_classes.IntraModeList
    angles: instance of class mol_classes.IntraModeList
    dihedrals: instance of class mol_classes.IntraModeList
    impropers: instance of class mol_classes.IntraModeList    
    """

    # Delete the appropriate atoms and interactions
    if len(delete) != 0:
        del_atom_ids = [ i for l in [molecules.atom_ids[idx] for idx,ID in enumerate(molecules.ids) if ID in delete] for i in l ]
        atoms.del_idx(list(np.array(del_atom_ids)-1),reassign_lammps_type=False)
        bonds.del_by_atom_ids(del_atom_ids,reassign_lammps_type=False,old2new_ids=atoms.old2new_ids)
        angles.del_by_atom_ids(del_atom_ids,reassign_lammps_type=False,old2new_ids=atoms.old2new_ids)
        dihedrals.del_by_atom_ids(del_atom_ids,reassign_lammps_type=False,old2new_ids=atoms.old2new_ids)
        impropers.del_by_atom_ids(del_atom_ids,reassign_lammps_type=False,old2new_ids=atoms.old2new_ids)

    # Adjust the mol numbers
    old2new_molids = { m:m_idx+1 for m_idx,m in enumerate(sorted(list(set(atoms.mol_id)))) }
    atoms.mol_id = np.array([ old2new_molids[m] for m in atoms.mol_id ])

    # Add the appropriate atoms and interactions
    mol_count = len(set(atoms.mol_id))
    atom_count = len(atoms.ids)
    interaction_types = ['Bonds','Angles','Dihedrals','Impropers']
    interaction_counts = {
        'Bonds': len(bonds.ids),
        'Angles': len(angles.ids),
        'Dihedrals': len(dihedrals.ids),
        'Impropers': len(impropers.ids)}
    new_atoms = {}
    new_interactions = { i:{} for i in interaction_types}

    # Loop through the "add" object, adding the new atoms and interactions to their respective dictionaries
    # Adding the items to dictionaries then converting to lists in one step is faster
    for k,v in add.items():

        # Create a dictionary to map this species' atom ids to the system's atom ids
        ref2sys_atomids = { a[0]:atom_count+a_idx+1 for a_idx,a in enumerate(msf[v[0]]['Atoms']) }

        # Add the new atoms, with appropriate system ids and adjusted coordinates
        avg_coords = np.mean(np.array([ a[4:] for a in msf[v[0]]['Atoms'] ]),axis=0)
        for a in msf[v[0]]['Atoms']:
            new_atoms[ref2sys_atomids[a[0]]] = [ mol_count+1, a[2], a[3], a[4]-avg_coords[0]+v[1][0], a[5]-avg_coords[1]+v[1][1], a[6]-avg_coords[2]+v[1][2]  ]
        mol_count += 1
        atom_count += len(msf[v[0]]['Atoms'])

        # Add the new bonding interactions, with the appropriate ids
        for interaction in interaction_types:
            if interaction not in msf[v[0]].keys(): continue
            for i in msf[v[0]][interaction]:
                new_interactions[interaction][interaction_counts[interaction]+1] = [i[1]] + [ref2sys_atomids[j] for j in i[2:]]
                interaction_counts[interaction] += 1

    # Add all the atoms and interactions to the respective objects
    akeys = sorted(list(new_atoms.keys()))
    bkeys = sorted(list(new_interactions['Bonds'].keys()))
    ankeys = sorted(list(new_interactions['Angles'].keys()))
    dkeys = sorted(list(new_interactions['Dihedrals'].keys()))
    ikeys = sorted(list(new_interactions['Impropers'].keys()))
    atoms.append(
        ids=akeys,
        mol_id=[new_atoms[k][0] for k in akeys],
        lammps_type=[new_atoms[k][1] for k in akeys],
        charge=[new_atoms[k][2] for k in akeys],
        x=[new_atoms[k][3] for k in akeys], y=[new_atoms[k][4] for  k in akeys], z=[new_atoms[k][5] for k in akeys])
    bonds.append(
        ids=bkeys,
        lammps_type=[new_interactions['Bonds'][bk][0] for bk in bkeys],
        atom_ids=[ new_interactions['Bonds'][bk][1:] for bk in bkeys ])
    angles.append(
        ids=ankeys,
        lammps_type=[new_interactions['Angles'][ank][0] for ank in ankeys],
        atom_ids=[ new_interactions['Angles'][ank][1:] for ank in ankeys ])
    dihedrals.append(
        ids=dkeys,
        lammps_type=[new_interactions['Dihedrals'][dk][0] for dk in dkeys],
        atom_ids=[ new_interactions['Dihedrals'][dk][1:] for dk in dkeys ])
    impropers.append(
        ids=ikeys,
        lammps_type=[new_interactions['Impropers'][ik][0] for ik in ikeys],
        atom_ids=[ new_interactions['Impropers'][ik][1:] for ik in ikeys ])

    # Adjust charges to nuetralize system
    atoms.charge = np.array([ i-np.sum(atoms.charge)/len(atoms.charge) for i in atoms.charge ])

    return atoms,bonds,angles,dihedrals,impropers


def find_dist_same(geo,box):
    """Calculates MIC distance between points.

    Calculates the minimum image convention (MIC) distance between points,
    as supplied in a single array (hence "same"). Distance is Minkowski
    distance of p=1.0.

    Parameters
    ----------
    geo: np.ndarray
        Nx3 array, where each row holds the x, y, and z coordinates.
    box: list of lists of floats
        [[ xmin, xmax ], ... ]

    Returns
    -------
    rs: np.ndarray
        NxN array, where an entry is the distance between original point
        of row index and original point of column index.
    """

    rs = np.zeros((geo.shape[0], geo.shape[0]))
    for i in range(3):
        dist = squareform(pdist(geo[:,i:i+1], 'minkowski', p=1.0))
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all():
            dist -= l * (dist > l2)
            dist = np.abs(dist)
        rs += dist**2
    rs = np.sqrt(rs)

    return rs


def get_progression(counts,times,selected_rxns,reaction_types,species,windowsize=1e10):
    """Converts history information into a progression DataFrame.

    Given the system history, this function returns a DataFrame
    containing the molecule concentrations and the number of times each
    reaction ha been selected over the past MDMC cycles.

    Parameters
    ----------
    counts: dict
        - counts[diffusion step][rxn step][species] = count
        - counts[int][int][str] = int
    times: dict
        - times[diffusion step][rxn step] = time
        - times[int][int] = float
    selected_rxns: dict
        - selected_rxns[diffusion step][rxn step][rxn num] = count
        - selected_rxns[int][int][int] = int
    reaction_types: list
        - list of reaction types 
    species: list
        - list of species
    window_size: int
        - number of previous cycles to keep in the progression object

    Returns
    -------
    progression: pandas.DataFrame
        - indices: mdmc cycles
        - columns: 'time', reactions, species
        - entries: time entry, or reaction count,
                   or species number fraction for each cycle
    mdmc_cycles: list
        - enries are tuples; (mdmc cycle, (diffusion step, rxn step))
    """

    # Create a list of relevant MDMC cycles, with each entry containing
    # the new MDMD cycle (0 indexed) number and the (step,rc)
    mdmc_cycles = [
        (num,tup) for num,tup in enumerate([ 
        (step,rc) for step in sorted(counts.keys()) for rc in sorted(counts[step].keys()) ])
    ]
    windowsize = np.min([windowsize,len(mdmc_cycles)])
    mdmc_cycles = mdmc_cycles[int(len(mdmc_cycles)-windowsize):]

    # Declare the progression columns and data objects
    index_ = [ i[0] for i in mdmc_cycles ]
    columns_ = ['time'] + reaction_types + species
    data_ = {}

    # Fill in the data
    data_['time'] = [ times[i[1][0]][i[1][1]] for i in mdmc_cycles ]
    for r in reaction_types:
        data_[r] = [ selected_rxns[i[1][0]][i[1][1]].count(r) for i in mdmc_cycles ]
    for s in species:
        data_[s] = [ counts[i[1][0]][i[1][1]][s] for i in mdmc_cycles ]

    # Create the progression DataFrame
    progression = pd.DataFrame( data=data_, columns=columns_, index=index_ )

    # Change from molecule counts to molecule number fraction
    progression.loc[:,species[0]:species[-1]] = progression.loc[:,species[0]:species[-1]].div(
        progression.loc[:,species[0]:species[-1]].sum(axis=1),axis=0)

    return progression,mdmc_cycles


def update_progression(progression,molecules,species,time,selected_rxns,windowsize_MDMCcycles=1e10):

    add_index = [progression.index[-1] + 1]
    add_columns = progression.columns
    add_data = {'time': [time]}
    for c in [_ for _ in progression.columns if type(_) == int]:
        add_data[c] = [selected_rxns.count(c)]
    for sp in species:
        add_data[sp] = [len([ 1 for _ in molecules.mol_types if _ == sp ])]
    add_df = pd.DataFrame( data=add_data, index=add_index, columns=add_columns )
    add_df.loc[:,species[0]:species[-1]] = add_df.loc[:,species[0]:species[-1]].div(add_df.loc[:,species[0]:species[-1]].sum(axis=1),axis=0)
    progression = pd.concat([progression,add_df])
    if progression.shape[0] > windowsize_MDMCcycles:
        progression = progression.drop(labels=progression.index[0],axis=0)

    return progression

def write_tracking_file(file_name,lines,driver='\"Driver name\"',time='\"datetime\"',step=None,new=False):

    parameter = 'a'
    file_type='Unknown file type'
    if new:
        parameter = 'w'
        if file_name.lower()[-4:] == 'conc':
            file_type = 'Concentration'
        if file_name.lower()[-3:] == 'log':
            file_type = 'Log'
        if file_name.lower()[-5:] == 'scale':
            file_type = 'Scaling'

    with open(file_name,parameter) as f:
        if new:
            f.write('# {} file created for tracking progress in a Hybrid MD/MC simulation.\n'.format(file_type))
            f.write('# Create with {} on {}\n'.format(driver,time))
        f.write('\n{}\n\nDiffusionStep {}\n'.format('-'*75,step))
        for l in lines:
            f.write(l)

    return


def get_PSSrxns(
        rxndata,rxnmatrix,rxnscaling,progression,
        windowsize_slope,windowsize_scalingpause,
        scalingcriteria_concentration_slope,scalingcriteria_concentration_cycles):
    """Determine the reactions that are in psuedo steady state.

    Given certain criteria and the system history, this function
    returns a list of reactions that are in psuedo steady state.
    
    Criteria:

    Parameters
    ----------
    rxnscaling: pandas.DataFrame
        - index: MDMC cycle number
        - columns: rxn nums
        - entries: scaling factors
    progression: pandas.DataFrame
        - index: mdmc cycle
        - columns: 'time', rxn num, or species
        - entries: time entry (float), or reaction count (int),
                   or species number fraction (float)
    rxndata: dict
        - contains information about the predescribed reactions
        - see hybridmdmc.parsers.parse_rxn_data
        - also holds raw rxn rate in the 'rate' key
    """

    # If the windowsize_slope exceeds the current number of MDMC cycles,
    # return an empty list. By definition, no reaction can be at PSS.
    if windowsize_slope > len(progression):
        return []

    # For each species, determine the number of cycles that have occured
    # for which the slope of the concentration has met the
    # scalingcriteria_concentration_slope criteria.
    cyclesofconstantconcentration = {}
    for column in [_ for _ in progression.columns if _ != 'time' and type(_) == str]:
        cyclesofconstantconcentration[column] = get_cyclesofconstantslope(
            progression,column,windowsize_slope,scalingcriteria_concentration_slope)

    # Loop over the reactions, checking each for PSS based on the
    # desired criteria.
    PSSrxns = []
    for rxnnum in [_ for _ in progression.columns if _ != 0 and type(_) != str]:

        # Create a list of all species involved in this reaction,
        # reactants and products.
        species_rxn = [_ for _ in rxnmatrix.columns if rxnmatrix.loc[rxnnum,_] != 0]

        # If all of the species for this reaction have an unchanging
        # unchanging slope of concentration, add this reaction to the
        # PSS list.
        if np.all(np.array([cyclesofconstantconcentration[s] for s in species_rxn]) >= scalingcriteria_concentration_cycles):
            PSSrxns += [rxnnum]
        
    return PSSrxns


def ratescaling_unscalerxns(rxnmatrix,rxnscaling,progression,PSSrxns,cycle=None):
    """Unscale reaction rates.

    Given certain criteria and the system history, this function
    unscales reaction rates. In the driver, the order of operations
    MUST be unscale THEN scale. Otherwise, scaling would never occur
    because at the start of the algorithm, all reactions are unscaled,
    and thus an unscaled reaction will always be selected prior to any
    scaling. If the order was scale then unscale, all reactions would
    be unscaled at the begining, unless null reaction is selected for
    all voxels.
    
    Criteria:
    1.

    Unscaling:
    1. If a scaled reaction is selected, all reactions connected to the
    reactants and products of that reaction are unscaled.
    2. If an unscaled reaction is selected, ALL reactions are unscaled.

    Parameters
    ----------
    """

    # By adding the cycle parameter, this function will
    # explicitly examine the correct row of rxnscaling. Otherwise, might
    # be buggy if the main script is edited and ratescaling_unscalerxns
    # assumes the last row in rxnscaling is the most current cycle.
    if cycle == None:
        cycle = rxnscaling.index[-1]

    # Loop over a list of the most recently selected reactions.
    for rxntype in [_ for _ in progression.columns if _ != 0 and type(_) != str and progression.loc[cycle,_] != 0]:

        # Unscale all reactions if an unscaled reaction that is not in
        # PSS was selected.
        if rxnscaling.loc[cycle,rxntype] == 1.0 and rxntype not in PSSrxns:
            # rxnscaling.loc[cycle,:] = 1.0
            rxnscaling.loc[cycle,:] *= 10.0
            rxnscaling[rxnscaling > 1.0] = 1.0
            return rxnscaling

    rxnscaling[rxnscaling > 1.0] = 1.0

    return rxnscaling


def ratescaling_scalerxns(
        rxndata,rxnmatrix,rxnscaling,progression,PSSrxns,
        scalingcriteria_rxnselection_count,
        windowsize_scalingpause,windowsize_rxnselection,
        scalingfactor_adjuster,scalingfactor_minimum,rxnlist='all'):
    """Scale reaction rates.

    Given certain criteria and the system history, this function
    returns scaling for reaction rates.
    
    Criteria:
    1. The rolling mean of the concentration of a reaction's reactants
    (and products?) is "unchanging" (deviation maximum to be provided
    by user) for a user-defined number of cycles.

    Scaling:
    1. Downscale reactions in each step by either a set value, OR by the
    product of a set value and the ratio of the raw reaction rate over
    the maximum raw reaction rate.
    2. Set a floor for reaction scaling. Can either be a set value, OR
    it can depend on the raw reaction rates, for instance never drop
    below the smallest raw reaction rate.
    3. Amount of scaling can be dependent on the number of MDMC cycles
    over which the specific reaction meets the scaling criteria.

    Parameters
    ----------
    rxnscaling: pandas.DataFrame
        - index: MDMC cycle number
        - columns: rxn nums
        - entries: scaling factors
    progression: pandas.DataFrame
        - index: mdmc cycle
        - columns: 'time', rxn num, or species
        - entries: time entry (float), or reaction count (int),
                   or species number fraction (float)
    rxndata: dict
        - contains information about the predescribed reactions
        - see hybridmdmc.parsers.parse_rxn_data
        - also holds raw rxn rate in the 'rate' key
    """

    # Create a dictionary to track the new reaction scales
    newscaling = {rxnnum:rxnscaling.loc[rxnscaling.index[-1],rxnnum] for rxnnum in rxnscaling.columns}

    # Reset the windowsize_rxnselection if it is greater than the length of
    # the progression df.
    if windowsize_rxnselection > len(progression):
        windowsize_rxnselection = len(progression)

    # Loop over the desired reactions, scaling each appropriately.
    if rxnlist == 'all':
        rxnlist = list(rxnscaling.columns)
    for rxnnum in rxnlist:

        # Check if this reaction has been (un)scaled within the last
        # "windowsize_pause" number of MDMC cycles. If it has, copy the
        # previous rate scale and continue.
        if get_cyclesofconstantscaling(rxnscaling,rxnnum) <= windowsize_scalingpause:
            newscaling[rxnnum] = rxnscaling.loc[rxnscaling.index[-1],rxnnum]
            continue

        # Check if this reaction is in PSS. If not, reset the rate scale
        # to 1.0 and continue.
        if rxnnum not in PSSrxns:
            newscaling[rxnnum] = 1.0
            continue

        # Check that this reaction has been selected at least
        # scalingcriteria_rxnselection_count times in the previous
        # windowsize_rxnselection steps. If not, reset tthe rate scale
        # and continue.
        if np.sum(progression.loc[progression.index[-windowsize_rxnselection]:,rxnnum]) < scalingcriteria_rxnselection_count:
            newscaling[rxnnum] = rxnscaling.loc[rxnscaling.index[-1],rxnnum]
            continue

        # Scale this reaction.
        newscaling[rxnnum] = rxnscaling.loc[rxnscaling.index[-1],rxnnum] * scalingfactor_adjuster
        
    # Add the new row to rxnscaling.
    rxnscaling.loc[rxnscaling.index[-1]+1] = [newscaling[rxnnum] for rxnnum in rxnscaling.columns]

    # Finally, adjust all scaling factors that are below the minimum
    # scaling factor back to the minimum scaling factor.
    rxnscaling[rxnscaling < scalingfactor_minimum] = scalingfactor_minimum

    return rxnscaling


def ratescaling_allscale(
        rxndata,rxnmatrix,rxnscaling,progression,PSSrxns,
        scalingcriteria_rxnselection_count,
        windowsize_scalingpause,windowsize_rxnselection,
        scalingfactor_adjuster,scalingfactor_minimum,
        rxnlist='all',cycle=None):

    # Handle defaults
    if cycle == None:
        cycle = rxnscaling.index[-1]
    if rxnlist == 'all':
        rxnlist = list(rxnscaling.columns)
    if windowsize_rxnselection > len(progression):
        windowsize_rxnselection = len(progression)

    # Add a row to rxnscaling
    rxnscaling.loc[cycle+1] = rxnscaling.loc[cycle]

    # Find the cyclesofconstantscaling for each rxn
    cyclesofconstantscaling = {_:get_cyclesofconstantscaling(rxnscaling,_) for _ in rxnscaling.columns}

    # Loop over a list of the most recently selected reactions.
    for rxntype in [_ for _ in progression.columns if _ != 0 and type(_) != str and progression.loc[cycle,_] != 0]:

        # Descale all reactions if nonPSS rxn was selected.
        if rxntype not in PSSrxns:
            rxnscaling.loc[cycle+1] *= scalingadjuster_descale
            rxnscaling[rxnscaling > 1.0] = 1.0
            rxnscaling[rxnscaling < scalingfactor_minimum] = scalingfactor_minimum
            return rxnscaling

    # Loop over rxnlist and scale each appropriately
    for rxn in rxnlist:
        isnnijnrtgnw = 1


    # Return rxnscaling, after adjusting factors outside the bounds
    rxnscaling[rxnscaling > 1.0] = 1.0
    rxnscaling[rxnscaling < scalingfactor_minimum] = scalingfactor_minimum
    return rxnscaling


def get_rxnmatrix(rxndata,masterspecies):
    
    rxndict = {s:{n:0 for n in rxndata.keys()} for s in masterspecies.keys()}
    for n in rxndata.keys():
        for s in rxndata[n]['reactant_molecules']:
            rxndict[s][n] -= 1
        for s in rxndata[n]['product_molecules']:
            rxndict[s][n] += 1

    return pd.DataFrame(data=rxndict,columns=sorted(list(masterspecies.keys())),index=sorted(list(rxndata.keys())))


def get_cyclesofconstantscaling(rxnscaling,rxnnum):

    # If the rxnscaling dataframe has no length, return a value of 0.
    if len(rxnscaling) == 0:
        return 0

    # Create an array holding the scaling factors of the reaction in
    # question, in reverse order.
    rxnscaling_reverse = np.array(rxnscaling.loc[:,rxnnum][::-1])

    # If all of the scaling values for this reaction are the same,
    # return the length of the rxnscaling dataframe.
    if len(set(rxnscaling_reverse)) == 1:
        return len(rxnscaling_reverse)

    # Otherwise, determine how many cycles have occured since the last
    # scaling or unscaling event (i.e. how many cycles have occurred
    # since the scaling factor changed).
    return np.argwhere(rxnscaling_reverse[1:] != rxnscaling_reverse[:-1])[0][0] + 1


def get_cyclesofconstantrollingmean(progression,column,windowsize_rollingmean,scalingcriteria_rollingmean_stddev):

    # If the progression dataframe has no length, return a value of 0.
    if len(progression) == 0:
        return 0

    # If the windowsize_rollingmean is equal to the length of the
    # progession dataframe, "1" needs to be manually returned (the rest
    # of the function won't appropriately handle this).
    if len(progression) == windowsize_rollingmean:
        return 1
    
    # Create an array holding the concentration, in reverse order.
    concentration_reverse = np.array(progression.loc[:,column][::-1])

    # Calculate the rolling mean using the provided window size, in
    # reverse order. This array will be shorter than the progression
    # dataframe by "windowsize_rollingmean."
    rollingmean_reverse = np.array([
        np.mean(concentration_reverse[idx:idx+windowsize_rollingmean])
        for idx in range(0,len(concentration_reverse)-windowsize_rollingmean)])

    # Calculate the standard deviation of the rolling mean, in reverse
    # order.
    stddev_rollingmean_reverse = np.array([
        np.std(rollingmean_reverse[:idx+1]) for idx in range(len(rollingmean_reverse))])

    # If all of the cycles satisfy the scalingcriteria_rollingmean,
    # return the length of stddev_rollingmean_reverse. Not catching this
    # results in an error in the next step.
    if np.all(stddev_rollingmean_reverse <= scalingcriteria_rollingmean_stddev):
        return len(stddev_rollingmean_reverse)

    # Otherwise, return the index of the first instance where the
    # standard deviation of the reverse rolling mean does NOT satisfy
    # scalingcriteria_rollingmean. This represents the number of cycles,
    # counting back from the current cycle, that satisfy the criteria.
    return np.argwhere(stddev_rollingmean_reverse > scalingcriteria_rollingmean_stddev)[0][0]


def get_cyclesofconstantslope(progression,column,windowsize_slope,scalingcriteria_concentration_slope,steps=1e99):

    # If the progression dataframe has no length, return a value of 0.
    if len(progression) == 0 or len(progression) < windowsize_slope:
        return 0

    # Adust the steps
    steps += windowsize_slope
    if steps > len(progression):
        steps = len(progression)

    # Create an array holding the concentration, in reverse order.
    concentration_reverse = np.array(progression.loc[:,column][-steps:][::-1])

    # Calculate the absolute value of the slopes using the provided
    # window size, in reverse order.
    slope_reverse = np.array([
        np.abs(linear_model.LinearRegression().fit(np.array(range(windowsize_slope)).reshape(-1,1),concentration_reverse[idx:idx+windowsize_slope]).coef_[0])
        for idx in range(0,len(concentration_reverse)-windowsize_slope+1)])

    # If all of the cycles satisfy the
    # scalingcriteria_concentration_slope, return the length of
    # concentration_reverse - windowsize_slope + 1. Not catching this
    # results in an error in the next step.
    if np.all(slope_reverse <= scalingcriteria_concentration_slope):
        return len(slope_reverse)

    # Otherwise, return the index of the first instance where the
    # slope of the reverse concentraiton does NOT satisfy
    # scalingcriteria_concentraiton_slope. This represents the number of
    # cycles, counting back from the current cycle, that satisfy the
    # criteria.
    return np.argwhere(slope_reverse > scalingcriteria_concentration_slope)[0][0]