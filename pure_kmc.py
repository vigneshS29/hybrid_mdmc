#/usr/bin/env python


import argparse, datetime
import numpy as np
import pandas as pd
from hybrid_mdmc.functions import get_PSSrxns,scalerxns


def main():

    parser = argparse.ArgumentParser(description='Run a toy model of a reaction network with KMC and reaction scaling.')

    parser.add_argument('name', type=str, help='Name of the output file.')

    parser.add_argument('-species',   type=str, default= 'A:200, B:0, C:0, D:0')
    parser.add_argument('-reactants', type=str, default= '1:A,   2:B,   3:C,   4:B,   5:D,   6:B')
    parser.add_argument('-products',  type=str, default= '1:B,   2:A,   3:B,   4:C,   5:B,   6:D')
    parser.add_argument('-rates',     type=str, default= '1:2e7, 2:2e7, 3:2e7, 4:2e7, 5:1e3, 6:2e4')

    parser.add_argument('-steps', type=int, default=4000)

    parser.add_argument('-windowsize_slope', type=int, default=8)
    parser.add_argument('-scalingcriteria_concentration_slope', type=float, default=0.2)
    parser.add_argument('-scalingcriteria_concentration_cycles', type=int, default=1)
    parser.add_argument('-windowsize_rxnselection', type=int, default=10)
    parser.add_argument('-scalingcriteria_rxnselection_count', type=int, default=2)
    parser.add_argument('-windowsize_scalingpause', type=int, default=2)
    parser.add_argument('-scalingfactor_adjuster', type=float, default=0.001)
    parser.add_argument('-scalingfactor_minimum', type=float, default=0.001)

    args = parser.parse_args()

    species = {_.split(':')[0].strip():float(_.split(':')[1]) for _ in args.species.split(',')}
    species_IDs = sorted(list(species.keys()))
    rates = {int(_.split(':')[0]):float(_.split(':')[1]) for _ in args.rates.split(',')}
    reaction_IDs = sorted(list(rates.keys()))
    reactants = {int(_.split(':')[0]):_.split(':')[1].strip() for _ in args.reactants.split(',')}
    products = {int(_.split(':')[0]):_.split(':')[1].strip() for _ in args.products.split(',')}
    progression = {sp:[species[sp]] for sp in species_IDs}
    progression.update({rxn:[0] for rxn in reaction_IDs})
    progression.update({'time':[0]})
    progression.update({0:[0]})
    progression = pd.DataFrame(
        progression,
        columns=['time']+species_IDs+[0]+reaction_IDs,
        index=[0])
    reaction_scaling = pd.DataFrame(
        {rxn:[1.0] for rxn in reaction_IDs},
        columns=reaction_IDs,
        index=[0])
    reaction_matrix = pd.DataFrame(
        {sp:[0 for _ in reaction_IDs] for sp in species_IDs},
        columns=species_IDs,
        index=reaction_IDs)
    for rxn in reaction_IDs:
        reaction_matrix.loc[rxn,reactants[rxn]] = -1
        reaction_matrix.loc[rxn,products[rxn]] = 1

    progression, reaction_scaling = conduct_pure_kmc(
        progression,
        reaction_matrix,
        reaction_scaling,
        args.windowsize_slope,
        args.windowsize_rxnselection,
        args.windowsize_scalingpause,
        args.scalingcriteria_concentration_slope,
        args.scalingcriteria_concentration_cycles,
        args.scalingcriteria_rxnselection_count,
        args.scalingfactor_adjuster,
        args.scalingfactor_minimum,
        reactants,
        products,
        reaction_IDs,
        rates,
        species_IDs,
        args.steps,
    )

    with open(args.name + '_KMCoutput.txt', 'w') as f:
        f.write('KMC output\nWritten {}\n\n'.format(datetime.datetime.now()))
        f.write('Progression\n')
        f.write(progression.to_string())
        f.write('\n\n')
        f.write('Reaction scaling\n')
        f.write(reaction_scaling.to_string())

    return


def pure_kmc_reaction_selection(probabilities, reaction_IDs):
    u2 = 0
    while u2 == 0:
        u2 = np.random.random()
    dt = -np.log(u2)/np.sum(probabilities)
    u1 = np.random.random()
    rxn_idx = np.argwhere(np.cumsum(probabilities)>=np.sum(probabilities)*u1)[0][0]
    selected_reaction_ID = reaction_IDs[rxn_idx]
    return selected_reaction_ID, dt


def conduct_pure_kmc(
        progression,
        reaction_matrix,
        reaction_scaling,
        windowsize_slope,
        windowsize_rxnselection,
        windowsize_scalingpause,
        scalingcriteria_concentration_slope,
        scalingcriteria_concentration_cycles,
        scalingcriteria_rxnselection_count,
        scalingfactor_adjuster,
        scalingfactor_minimum,
        reactants,
        products,
        reaction_IDs,
        rates,
        species_IDs,
        total_steps,
        ):
    
    for step in range(total_steps):

        PSSrxns = get_PSSrxns(
            reaction_matrix,
            reaction_scaling,
            progression,
            windowsize_slope,
            windowsize_rxnselection,
            scalingcriteria_concentration_slope,
            scalingcriteria_concentration_cycles,
            scalingcriteria_rxnselection_count,
        )

        reaction_scaling = scalerxns(
            reaction_scaling,
            PSSrxns,
            windowsize_scalingpause,
            scalingfactor_adjuster,
            scalingfactor_minimum,
            rxnlist='all',
        )

        probabilities = np.array([rates[_]*progression.loc[step,reactants[_]]*reaction_scaling.loc[step+1,_] for _ in reaction_IDs])
        selected_reaction_ID, dt = pure_kmc_reaction_selection(probabilities, reaction_IDs)
        progression.loc[step+1] = [progression.loc[step,'time']+dt] + [progression.loc[step,_] for _ in species_IDs] + [0] + [0 for _ in reaction_IDs]
        progression.loc[step+1,reactants[selected_reaction_ID]] -= 1
        progression.loc[step+1,products[selected_reaction_ID]] += 1
        progression.loc[step+1,selected_reaction_ID] += 1

    return progression, reaction_scaling

if __name__ == '__main__':
    main()