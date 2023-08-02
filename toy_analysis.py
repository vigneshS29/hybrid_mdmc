#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
sys.path.append('Users/dgilley/bin/')
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D
from copy import deepcopy
from sklearn import linear_model
from plotting import formatmpl

print(sys.path)

#%%
from Users.dgilley.bin.hybrid_mdmc.parsers import parse_concentration,parse_scale,parse_rxndf,parse_msf
from hybrid_mdmc.functions import get_progression


#%%
print(sys.path)

# In[6]:


def histo_getbins(times, num_of_bins=100, tmin=None, tmax=None):
    if not tmax:
        tmax = np.min([np.max(_) for _ in times])
    if not tmin:
        tmin = np.max([np.min(_) for _ in times])
    binsize = (tmax - tmin)/num_of_bins
    return np.array([[tmin + i*binsize, tmin + (i+1)*binsize] for i in range(num_of_bins)]), tmin, tmax


def histo_bytime(times, datas, num_of_bins=100, tmin=None, tmax=None):
    bins, tmin, tmax = histo_getbins(
        times, num_of_bins=num_of_bins, tmin=tmin, tmax=tmax)
    binned_data = {_: [] for _ in range(num_of_bins)}
    for data_idx in range(len(times)):
        for old_idx, t in enumerate(times[data_idx]):
            new_idx = int((t-tmin)/(bins[0][1]-bins[0][0]))
            if new_idx >= len(bins):
                continue
            try:
                binned_data[new_idx].append(datas[data_idx][old_idx])
            except:
                print(binned_data[new_idx], new_idx,
                      len(datas[data_idx]), old_idx)
                binned_data[new_idx].append(datas[data_idx][old_idx])
    bins = [bins[idx] for idx, _ in binned_data.items() if len(_)]
    avg = [np.mean(binned_data[_])
           for _ in range(num_of_bins) if len(binned_data[_])]
    std = [np.std(binned_data[_])
           for _ in range(num_of_bins) if len(binned_data[_])]
    return bins, avg, std


def calc_Eyring(A, Ea, T):
    # A units determine the returned units, usually 1/s
    # Ea in kcal/mol
    # T in K
    R = 0.00198588  # kcal / mol K
    return A*T**(1.0)*np.exp(-Ea/T/R)


def calc_Ea(A, k, T):
    # A in same units as k; usually 1/s
    # k in same units as A; usually 1/s
    # T in K
    # Ea in kcal/mol
    R = 0.00198588  # kcal / mol K
    return -R*T*np.log(k/A/T)


def plot_counts(
        meta, group, species, color_groups, linestyle_groups,
        runs='all', tmax=False, num_of_bins=100, equilibrium=False):

    if runs == 'all':
        runs = meta.loc['Runs', group]
    if not tmax:
        tmax = np.min([np.sum(progression[group][_].loc[:, 'time'])
                       for _ in runs])

    t, a, s = histo_bytime(
        [np.cumsum(progression[group][_].loc[:, 'time']) for _ in runs],
        [progression[group][_].loc[:, species] for _ in runs],
        num_of_bins=num_of_bins, tmax=tmax)
    time = np.array([0] + [np.mean(_) for _ in t])
    avg = [meta.loc['Starting {}'.format(species), group]] + a
    std = [0] + s

    plt.plot(
        time,
        avg,
        color=color_groups[group], linestyle=linestyle_groups[group], linewidth=4, alpha=0.8
    )

    plt.fill_between(
        time,
        np.array(avg)-np.array(std),
        np.array(avg)+np.array(std),
        color=color_groups[group], alpha=0.1
    )

    if equilibrium:
        plt.plot(
            [0, xlims[1]*1.5],
            [equilibrium, equilibrium],
            color=color_groups[group], linestyle='--', linewidth=4, alpha=1.0
        )

    return


def plot_counts_bystep(
        meta, group, species, color_groups, linestyle_groups,
        runs='all', smax=False, num_of_bins=100, equilibrium=False):

    if runs == 'all':
        runs = meta.loc['Runs', group]
    if not smax:
        smax = np.min([np.sum(progression[group][_].index) for _ in runs])

    s, a, s = histo_bytime(
        [progression[group][_].index for _ in runs],
        [progression[group][_].loc[:, species] for _ in runs],
        num_of_bins=num_of_bins, tmax=tmax)
    step = np.array([0] + [np.mean(_) for _ in s])
    avg = [meta.loc['Starting {}'.format(species), group]] + a
    std = [0] + s

    plt.plot(
        step,
        avg,
        color=color_groups[group], linestyle=linestyle_groups[group], linewidth=4, alpha=0.8
    )

    plt.fill_between(
        step,
        np.array(avg)-np.array(std),
        np.array(avg)+np.array(std),
        color=color_groups[group], alpha=0.1
    )

    if equilibrium:
        plt.plot(
            [0, xlims[1]*1.5],
            [equilibrium, equilibrium],
            color=color_groups[group], linestyle='--', linewidth=4, alpha=1.0
        )

    return


def plot_rxnselection(
        meta, group, rxn, colors, linestyles,
        runs='all', tmax=False, num_of_bins=100, equilibrium=False):

    if runs == 'all':
        runs = meta.loc['Runs', group]
    if not tmax:
        tmax = np.min([np.sum(progression[group][_].loc[:, 'time'])
                       for _ in runs])

    t, a, s = histo_bytime(
        [np.cumsum(progression[group][_].loc[:, 'time']) for _ in runs],
        [np.cumsum(progression[group][_].loc[:, rxn]) for _ in runs],
        num_of_bins=num_of_bins, tmax=tmax)
    time = np.array([0] + [np.mean(_) for _ in t])
    avg = [0] + a
    std = [0] + s

    plt.plot(
        time,
        avg,
        color=colors[rxn], linestyle=linestyles[group], linewidth=4, alpha=0.5
    )

    # plt.fill_between(
    #    time,
    #    np.array(avg)-np.array(std),
    #    np.array(avg)+np.array(std),
    #    color=colors[rxn],alpha=0.1
    # )

    return


def plot_rxnscaling(
        meta, scaling, progression, group, rxn, color_groups, linestyle_rxns,
        runs='all', tmax=False, num_of_bins=100, equilibrium=False):

    if runs == 'all':
        runs = meta.loc['Runs', group]
    if not tmax:
        tmax = np.min([np.sum(progression[group][_].loc[:, 'time'])
                       for _ in runs])

    time, avg, std = histo_bytime(
        [np.cumsum(progression[group][_].loc[:, 'time']) for _ in runs],
        [scaling[group][_].loc[:, rxn] for _ in runs],
        num_of_bins=num_of_bins, tmax=tmax)

    time = [np.mean(_) for _ in time]

    plt.semilogy(
        time,
        avg,
        color=color_groups[group], linestyle=linestyle_rxns[rxn], linewidth=4, alpha=0.8
    )

    # plt.fill_between(
    #    np.array(time).reshape(-1),
    #    np.array(avg)-np.array(std),
    #    np.array(avg)+np.array(std),
    #    color=color_rxns[rxn],alpha=0.1
    # )

    return


def calc_dCdt_direct(
        count, time,
        window=40):

    if window % 2 != 0:
        window += 1

    dCdt = np.array([
        linear_model.LinearRegression().fit(
            np.array(
                np.cumsum(time[int(idx-window/2):int(idx+window/2)])).reshape(-1, 1),
            np.array(count[int(idx-window/2):int(idx+window/2)]).reshape(-1, 1)
        ).coef_[0][0]

        for idx in range(int(window/2), int(len(count)-window/2-1))])

    return [dCdt, count[int(window/2):int(len(count)-window/2-1)], np.cumsum(time[int(window/2):int(len(count)-window/2-1)])]


def calc_and_bin_dCdt_direct(
        meta, progression, group, species,
        window=40, num_of_bins=100):

    runs = meta.loc['Runs', group]

    dCdt = {
        run: calc_dCdt_direct(
            progression[group][run].loc[:, sp],
            progression[group][run].loc[:, 'time'],
            window=window)
        for run in meta.loc['Runs', group]}
    time, mean, std = histo_bytime(
        [dCdt[_][2] for _ in runs],
        [dCdt[_][0] for _ in runs],
        tmax=np.min([np.sum(progression[group][_].loc[:, 'time'])
                     for _ in meta.loc['Runs', group]]),
        num_of_bins=num_of_bins)
    time = np.array([np.mean(_) for _ in time])

    return time, mean, std


def calc_dCdt_fromelrates(
        meta, progression, group, run, species,
        A=1e12, T=188):

    conc = {_: progression[group][run][_].to_numpy()
            for _ in progression[group][run].columns}

    if group in ['toy1', 'toy2', 'toy3', 'toy4', 'toy5', 'toy6']:
        k1 = calc_Eyring(A, meta.loc['Ea Rxn 1 (kcal/mol)', group], T)
        k2 = calc_Eyring(A, meta.loc['Ea Rxn 2 (kcal/mol)', group], T)
        if species == 'A':
            return -(1/2)*k1*(conc['A']**2) + 2*k2*(conc['AA'])
        if species == 'AA':
            return k1*(conc['A']**2) - k2*(conc['AA'])

    if group in ['toy22', 'toy23', 'toy24', 'toy25']:
        k1 = calc_Eyring(A, meta.loc['Ea Rxn 1 (kcal/mol)', group], T)
        k2 = calc_Eyring(A, meta.loc['Ea Rxn 2 (kcal/mol)', group], T)
        k3 = calc_Eyring(A, meta.loc['Ea Rxn 3 (kcal/mol)', group], T)
        k4 = calc_Eyring(A, meta.loc['Ea Rxn 4 (kcal/mol)', group], T)
        if species == 'A':
            return -k1*(conc['A']) + k2*(conc['B'])
        if species == 'B':
            return k1*(conc['A']) - k2*(conc['B']) - k3*(conc['B']) + k4*(conc['C'])
        if species == 'C':
            return k3*(conc['B']) - k4*(conc['C'])


def calc_and_bin_dCdt_fromelrates(
        meta, progression, group, species,
        num_of_bins=100, A=1e12, T=188):

    runs = meta.loc['Runs', group]

    dCdt = {
        _: calc_dCdt_fromelrates(
            meta, progression, group, _, species, A=A, T=T)
        for _ in runs
    }

    time, mean, std = histo_bytime(
        [np.cumsum(progression[group][_].loc[:, 'time']) for _ in runs],
        [dCdt[_] for _ in runs],
        tmax=np.min([np.sum(progression[group][_].loc[:, 'time'])
                     for _ in runs]),
        num_of_bins=num_of_bins)
    time = np.array([np.mean(_) for _ in time])

    return time, mean, std


def get_rate_coeff(meta, group, A=1e12, T=188):

    if group in ['toy1', 'toy2', 'toy3', 'toy4', 'toy5', 'toy6']:
        k1 = calc_Eyring(A, meta.loc['Ea Rxn 1 (kcal/mol)', group], T)
        k2 = calc_Eyring(A, meta.loc['Ea Rxn 2 (kcal/mol)', group], T)
        return np.array([
            [-k1,    k2,  0],
            [k1, -k2-k3, k4],
            [0,    k3, -k4]])

    if group in ['toy{}'.format(_) for _ in range(22, 46)]:
        k1 = calc_Eyring(A, meta.loc['Ea Rxn 1 (kcal/mol)', group], T)
        k2 = calc_Eyring(A, meta.loc['Ea Rxn 2 (kcal/mol)', group], T)
        k3 = calc_Eyring(A, meta.loc['Ea Rxn 3 (kcal/mol)', group], T)
        k4 = calc_Eyring(A, meta.loc['Ea Rxn 4 (kcal/mol)', group], T)
        return np.array([
            [-k1,    k2,  0],
            [k1, -k2-k3, k4],
            [0,    k3, -k4]])


def get_X0(meta, group):

    if group in ['toy{}'.format(_) for _ in range(22, 46)]:
        return np.array([
            [meta.loc['Starting A', group]],
            [meta.loc['Starting B', group]],
            [meta.loc['Starting C', group]]])

    if group in ['toy1', 'toy2', 'toy3', 'toy4', 'toy5', 'toy6', 'toy10', 'toy11', 'toy12']:
        return np.array([
            [meta.loc['Starting A', group]],
            [meta.loc['Starting AA', group]],
        ])


def solve_rateODEs(
        meta, group, time,
        A=1e12, T=188):

    if group in ['toy{}'.format(_) for _ in range(22, 46)]:

        # Declare starting concentrations
        X0 = get_X0(meta, group)

        # Create matrix to hold the rate coefficients
        # dXdt = (rate_coeff)(X)
        # X: molecule number of each species
        rate_coeff = get_rate_coeff(meta, group, A=A, T=T)

        # Assume solution is X = K*exp(lambda*t)
        # Then, dXdt = lambda*K*exp(lambda*t) = rate_coeff*K*exp(lambda*t)
        # rate_coeff*K = lambda*K
        # For an NxN rate_coeff matrix there are up to N linearly independent
        # eigenvalues/eigenvectors that solve the above equation
        # Solve eigenvalue problem
        lam, K = np.linalg.eig(rate_coeff)

        # The general solution to the coupled ODE is a linear combination of the proposed solution
        # X = sum(ci*Ki*exp(lambdai*t))
        # To fine the ci, insert initial values of X (t = 0, exp(lambda*0) = 1)
        # X0 = KC
        # Solve for C with Guassian Elimination
        C = np.linalg.solve(K, X0)

        # Final solution is X = sum(ci*Ki*exp(lambdai*t))
        X = np.array([
            np.sum([C[_][0]*K[r][_]*np.exp(lam[_]*time)
                    for _ in range(len(X0))], axis=0)
            for r in range(len(X0))])

        return X

    if group in ['toy16', 'toy17', 'toy18']:
        k1 = calc_Eyring(A, meta.loc['Ea Rxn 1 (kcal/mol)', group], T)
        k2 = calc_Eyring(A, meta.loc['Ea Rxn 2 (kcal/mol)', group], T)
        a = 1
        b = k2/2
        c = 2*k1
        m1 = (-b+np.sqrt(b**2-4*c)) / (2)
        m2 = (-b+np.sqrt(b**2+4*c)) / (2)
        A0 = meta.loc['Starting A', group]
        AA0 = meta.loc['Starting AA', group]
        c2 = (-k1*A0**2 + k2*AA0 - m1*A0) / (m2-m1)
        c1 = A0 - c2
        Anum = c1*np.exp(m1*time) + c2*np.exp(m2*time)
        AAnum = (A0 + 2*AA0 - Anum) / (2)

        return np.array([Anum, AAnum])

    if group in ['toy1', 'toy2', 'toy3', 'toy4', 'toy5', 'toy6', 'toy10', 'toy11', 'toy12']:
        k1 = calc_Eyring(A, meta.loc['Ea Rxn 1 (kcal/mol)', group], T)
        k2 = calc_Eyring(A, meta.loc['Ea Rxn 2 (kcal/mol)', group], T)

        def dCdt(t, C):
            return [
                -2*k1*C[0]**2 + 2*k2*C[1],
                k1*C[0]**2 - k2*C[1]
            ]
        C0 = get_X0(meta, group).reshape(-1)
        sol = scipy.integrate.solve_ivp(
            dCdt, [time[0], time[-1]], C0, t_eval=time)
        return np.array([sol.y[0], sol.y[1]])


def calc_slope(time, count, window=10):
    #slope = [(count[idx]-count[idx-1]) / (time[idx]-time[idx-1]) for idx in range(1,len(count))]
    slope = np.array([
        linear_model.LinearRegression().fit(
            np.array(range(window)).reshape(-1, 1),
            count[idx-window:idx]
        ).coef_[0]
        for idx in range(window, len(count))
    ])
    return time[:len(slope)], slope


def calc_runningmean(x_old, y_old, window=10):
    x_new = np.array([np.mean(x_old[idx-window:idx])
                      for idx in range(window, len(x_old))])
    y_new = np.array([np.mean(y_old[idx-window:idx])
                      for idx in range(window, len(y_old))])
    return x_new, y_new


def calc_slope_runningmean(time, count, window_slope=10, window_rm=10):
    time, slope = calc_slope(time, count, window=window_slope)
    return calc_runningmean(time, slope, window=window_rm)


# # Excel Sheet
# Parameter information is stored in an excel sheet.
# 
# # Analysis
# All of the following items should be analyzed. This script should be written such that replicates of the same parameter sets should be grouped together. The average group behavior should be determined by histogramming on a time basis.
# 
# 1. Species counts
# 2. Reaction selection
# 3. Reaction scaling
# 4. Number of total diffusion steps, total reaction steps, and average reaction step per diffusion step
# 5. dC/dt for a species type
# 
# # Parameters
# - Runs: runs that share a parameter set (i.e. replicates)
# - Temperature: temperature of simulation (K)
# - RelaxationTime: length of the relaxation MD run (fs)
# - DiffusionTime: length of the diffusion MD run (fs)
# - DiffusionCutoff: maximum distance of separation between reactants (?) (Angstrom?)
# - ChangeThreshold: fraction of starting number of molecules to be deleted before the next diffusion step
# - Criteria_StdDev: maximum standard deviation of the rolling mean of the species number fraction to consider the species concentration to be "stagnant"
# - Criteria_Cycles: number of MDMC cycles that a species concentration must be stagnant before its reactions can be scaled
# - Window_Mean: number of MDMC cycles to use when calculating rolling means of concentrations
# - Window_Pause: number of MDMC cycles to wait between unscaling or scaling a reaction and allowing it to be scaled again
# - Scaling_Adjuster: quantity by which scaling factors are multiplied to (further) scale reactions
# - Scaling_Minimum: minimum allowed scaling factor

# In[3]:


groups = []
#groups += ['toy1','toy2','toy3']
#groups += ['toy4','toy5','toy6']
#groups += ['toy7','toy8','toy9']
#groups += ['toy10','toy11','toy12']
#groups += ['toy13','toy14','toy12']
#groups += ['toy13','toy14']
#groups += ['toy16','toy17','toy18']
#groups += ['toy19','toy20','toy21']
#groups += ['toy22','toy23','toy24','toy25']
#groups += ['toy26','toy27','toy28','toy29','toy30']
#groups += ['toy31','toy32','toy33','toy34','toy35']
#groups += ['toy36','toy37','toy38','toy39','toy40']
groups += ['toy41','toy42','toy43','toy44','toy45']
equilibrium = {
    'AA':{
         'toy1':98.2,  'toy2':97.4,  'toy3':94.8,
         'toy4':98.2,  'toy5':97.4,  'toy6':94.8,
         'toy7':98.2,  'toy8':97.4,  'toy9':94.8,
        'toy10':98.2, 'toy11':97.4, 'toy12':94.8,
        'toy13':   5, 'toy14':  50,
        'toy16':9.5,'toy17':100.0,'toy18':190.5,
        'toy19':9.5,'toy20':100.0,'toy21':190.5
    }
}

# If k' = n_i*k
equilibrium['AA'].update({
    #'toy1':98,'toy2':97,'toy3':95,
    #'toy4':97.5,'toy5':96.3,'toy6':92.7,
    #'toy7':97.5,'toy8':96.3,'toy9':92.7,
    #'toy10':97.5,'toy11':96.3,'toy12':92.7,
    #'toy12':92.7,'toy13':2.5,'toy14':38.1,
    }
)

meta = pd.read_excel('ToyParameters.xlsx',index_col=0)
meta.drop(columns=[g for g in meta.columns if g not in groups],inplace=True)
for g in groups:
    try:
        meta.loc['Runs',g] = [int(_) for _ in meta.loc['Runs',g].split(',')]
    except:
        meta.loc['Runs',g] = [int(meta.loc['Runs',g])]
#meta.loc['Runs','toy31'] = [1,3,4,5,6]
display(meta)


# In[4]:


progression = {group:{} for group in meta.columns}
scale = {group:{} for group in meta.columns}
for group in meta.columns:
    reaction_types = [0] + list(parse_rxndf(group+'.rxndf').keys())
    species = list(parse_msf(group+'.msf').keys())
    
    for run in meta.loc['Runs',group]:
        counts,times,selected_rxns = parse_concentration(group+'-{}.concentration'.format(run))
        progression[group][run] = get_progression(counts,times,selected_rxns,reaction_types,species)[0]
        scale[group][run] = parse_scale(group+'-{}.scale'.format(run))[0]


# In[9]:


A = 1e12
T = 188 # K
color_groups = {_:'#858585' for _ in meta.columns}
color_groups.update({
     'toy1':'#2431cb', 'toy2':'#f1ba7b', 'toy3':'#14d7a8',
     'toy4':'#2431cb', 'toy5':'#f1ba7b', 'toy6':'#14d7a8',
     'toy7':'#2431cb', 'toy8':'#f1ba7b', 'toy9':'#14d7a8',
    'toy10':'#2431cb','toy11':'#f1ba7b','toy12':'#14d7a8',
    'toy13':'#ff7676', 'toy14':'#e476ff',
    'toy16':'#2431cb', 'toy17':'#f1ba7b', 'toy18':'#14d7a8',
    'toy19':'#2431cb', 'toy20':'#f1ba7b', 'toy21':'#14d7a8',
    'toy22':'#cb2424', 'toy23':'#cbc824', 'toy24':'#24cb38', 'toy25':'#2436cb',
    'toy26':'#cb2424', 'toy27':'#cbc824', 'toy28':'#2436cb', 'toy29':'#00991e', 'toy30':'#8d0099',
    'toy31':'#cb2424', 'toy32':'#cbc824', 'toy33':'#2436cb', 'toy34':'#00991e', 'toy35':'#8d0099',
    'toy36':'#cb2424', 'toy37':'#cbc824', 'toy38':'#2436cb', 'toy39':'#00991e', 'toy40':'#8d0099',
    'toy41':'#cb2424', 'toy42':'#cbc824', 'toy43':'#2436cb', 'toy44':'#00991e', 'toy45':'#8d0099'


})

#color_groups.update({
#    'toy12':'#2431cb','toy13':'#14d7a8', 'toy14':'#f1ba7b',
#})

color_species = {_:'#858585' for _ in species}
color_species.update({
     'A':'#990000','AA':'#998b00','B':'#000099','C':'#009990'
})

linestyle_groups = {_:'-' for _ in meta.columns}
linestyle_groups.update({
     'toy1':'-', 'toy2':'-', 'toy3':'-',
     'toy4':'-', 'toy5':'-', 'toy6':'-',
     'toy7':':', 'toy8':':', 'toy9':':',
    'toy10':'-','toy11':'-','toy12':'-',
    'toy13':'-', 'toy14':'-',
    'toy16':'-','toy17':'-','toy18':'-',
    'toy19':'-','toy20':'-','toy21':'-'
})
color_rxns = {_:'#858585' for _ in reaction_types}
color_rxns.update({1:'#ff3d3d',2:'#ffe53d',3:'#3d43ff',4:'#df3dff'})
linestyle_rxns = {_:'-' for _ in meta.columns}
linestyle_rxns.update({
    0:'-',
    1:'-',2:'--',
    3:'-.',4:':'
})
label_map = {_:_ for _ in meta.columns}
label_map.update({
    'toy22':'kf/ks=1', 'toy23':'kf/ks=10','toy24':'kf/ks=100','toy25':'kf/ks=1000',
    0:'null',1:'fast,f',2:'fast,r',3:'slow,f',4:'slow,r',
})
label_map.update({
    g:'kf/ks={:<5.0f}'.format(calc_Eyring(A,meta.loc['Ea Rxn 1 (kcal/mol)',g],T)/calc_Eyring(A,meta.loc['Ea Rxn 3 (kcal/mol)',g],T))
    for g in meta.columns})
groups_str = '-'.join([i.split()[0] for i in meta.columns])
tmax = np.min(
    [np.min([np.sum(progression[group][run].loc[:,'time']) for run in meta.loc['Runs',group]])
     for group in meta.columns
    ]
)
# tmax = 0.7e-3
# xlims = [0-tmax/50,tmax-tmax/50]

#%%




# In[14]:


# Plot the counts of a certain species type
species = 'B'
sidx = 1

# Create figure
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='# of {} Molecules'.format(species)
)

# Display legend
handles = [
    mp.Patch(color=color_groups[g],alpha=0.7)
    for g in meta.columns] + [
    Line2D([0],[0],color='#000000',linestyle='-',alpha=0.7,linewidth=4),
    Line2D([0],[0],color='#000000',linestyle='--',alpha=0.7,linewidth=4)]
labels = [label_map[g] for g in meta.columns] + ['hybridMDMC','ODEs']
plt.grid()
plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.4,1.03))

# Plot
for group in meta.columns:
    plot_counts(
        meta,group,species,color_groups,linestyle_groups,
        runs='all',tmax=tmax,num_of_bins=300,
        equilibrium=False,
    )
    time_ = np.linspace(0,np.min([np.sum(progression[group][_].loc[:,'time']) for _ in meta.loc['Runs',group]]),num=1000)
    time_ = np.linspace(0,tmax,num=1000)
    X = solve_rateODEs(
            meta,group,time_,
            A=1e12,T=188)
    plt.plot(time_,X[sidx],color=color_groups[group],linestyle='--',linewidth=4,alpha=0.8)
plt.plot([0],[meta.loc['Starting {}'.format(species),group]],color='#000000',marker='s',ms=10)
    
# Save figure
xbound = 0.2e-4
plt.ylim(-5,5)
#plt.xlim(-xbound/10,xbound)
#plt.xlim(xlims)
plt.savefig('MoleculeCount{}_{}.pdf'.format(species,groups_str),bbox_inches='tight')
#plt.savefig('MoleculeCountZ{}.pdf'.format(groups_str),bbox_inches='tight')


# In[8]:


species = 'B'

running_mean_slope = {
    group:{
        #run:calc_slope_runningmean(
        run:calc_slope(
            np.cumsum(progression[group][run].loc[:,'time']),
            progression[group][run].loc[:,species],
            window=15
        )
        for run in meta.loc['Runs',group]
    }
    for group in meta.columns
}


# In[ ]:


# Plot the slopes

# Create figure
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='Slope of {} Per Time'.format(species)
)

# Display legend
handles = [
    mp.Patch(color=color_groups[g],alpha=0.7)
    for g in meta.columns]
labels = [label_map[g] for g in meta.columns]
plt.grid()
plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.4,1.03))

# Plot
for group in ['toy34','toy35']:#meta.columns:
    for ridx,run in enumerate(meta.loc['Runs',group]):
        plt.plot(
            running_mean_slope[group][run][0],
            running_mean_slope[group][run][1],
            color=color_groups[group],linewidth=2,alpha=0.3)
    
# Save figure
plt.ylim(-2.5,2.5)
plt.savefig('MoleculeSlope{}_{}.pdf'.format(species,groups_str),bbox_inches='tight')


# In[37]:


# Plot the cumulative reaction selections
window = 15

# Create figure
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='Rxn Selection'.format(species)
)

# Display legend
handles = [
    mp.Patch(color=color_groups[g],alpha=0.7)
    for g in meta.columns] + [
    Line2D([0],[0],color='#000000',linestyle=linestyle_rxns[_],alpha=0.7,linewidth=4) for _ in plot_rxns]
labels = [label_map[g] for g in meta.columns] + [rxn_map[_] for _ in plot_rxns]
plt.grid()
plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.4,1.03))

# Plot
for group in ['toy40']:#meta.columns:
    for rxn in [1,2,3,4]:
        for run in meta.loc['Runs',group]:
            #time = [_ for idx,_ in enumerate(np.cumsum(progression[group][run].loc[:,'time']),start=window,step=window)]
            #count = [np.sum(progression[group][run].loc[idx-window:idx,rxn]) for idx,_ in enumerate(progression[group][run].loc[:,rxn],start=window,step=window)]
            time = [
                np.sum(progression[group][run].loc[:idx,'time'])
                for idx in range(window,len(progression[group][run]),window)
            ]
            count = [
                np.sum(progression[group][run].loc[idx-window:idx,rxn])
                for idx in range(window,len(progression[group][run]),window)
            ]
            plt.plot(
                time,
                count,
                color=color_groups[group],linestyle=linestyle_rxns[rxn],linewidth=2,alpha=0.3
            )
    
# Save figure
#plt.xlim(0,1e6)
plt.savefig('ReactionSelection{}.pdf'.format(groups_str),bbox_inches='tight')


# In[15]:


# Plot the cumulative reaction selections
plot_rxns = [1,2,3,4]
rxn_map = {
    1:'A \u2192 B',
    2:'B \u2192 A',
    3:'B \u2192 C',
    4:'C \u2192 B',
}

# Create figure
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='Reaction Scaling'.format(species)
)

# Display legend
# Display legend
handles = [
    mp.Patch(color=color_groups[g],alpha=0.7)
    for g in meta.columns] + [
    Line2D([0],[0],color='#000000',linestyle=linestyle_rxns[_],alpha=0.7,linewidth=4) for _ in plot_rxns]
labels = [label_map[g] for g in meta.columns] + [rxn_map[_] for _ in plot_rxns]
plt.grid()
plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.4,1.03))

# Plot
for group in meta.columns:
    for rxn in plot_rxns:
        plot_rxnscaling(
            meta,scale,progression,group,rxn,color_groups,linestyle_rxns,
            runs='all',tmax=False,num_of_bins=100,equilibrium=False)
    
# Save figure
plt.savefig('ReactionScaling{}.pdf'.format(groups_str),bbox_inches='tight')


# In[113]:


window = 50
num_of_bins = 100

dCdt_direct_mean = {_:{} for _ in meta.columns}
dCdt_direct_std =  {_:{} for _ in meta.columns}
dCdt_direct_time = {_:{} for _ in meta.columns}

for group in meta.columns:
    for sp in ['A','B','C']:
        dCdt_direct_time[group][sp],dCdt_direct_mean[group][sp],dCdt_direct_std[group][sp] = calc_and_bin_dCdt_direct(
            meta,progression,group,species,
            window=window,num_of_bins=num_of_bins)


# In[ ]:


plot_species = ['A','B','C']

dCdt_rates_mean =  {_:{} for _ in meta.columns}
dCdt_rates_std =   {_:{} for _ in meta.columns}
dCdt_rates_time =  {_:{} for _ in meta.columns}

A = 1e12 # 1/s
T = 188  # K

for group in meta.columns:
    for sp in plot_species:
        dCdt_rates_time[group][sp],dCdt_rates_mean[group][sp],dCdt_rates_std[group][sp] = calc_and_bin_dCdt_fromelrates(
            meta,progression,group,sp,
            num_of_bins=50,A=A,T=T)


# In[ ]:


# 
plot_species = ['A','B','C']
plot_species = ['C']
group = 'toy22'

# Create figure
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='dC/dt'
)
plt.title('{}'.format(label_map[group]))

# Display legend
handles = [
    Line2D([0],[0],color='#000000',linestyle='-',linewidth=4,alpha=0.8),
    Line2D([0],[0],color='#000000',linestyle='--',linewidth=4,alpha=0.8),
    ] + [
    mp.Patch(color=color_species[_],alpha=0.7) for _ in plot_species
    ]
labels = [
    'direct','from rates'] + plot_species
plt.grid()
plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.35,1.03))

for species in plot_species:
    plt.plot(
        dCdt_direct_time[group][species],
        dCdt_direct_mean[group][species],
        color=color_species[species],linestyle='-',linewidth=4,alpha=0.6
    )
    plt.plot(
        dCdt_rates_time[group][species],
        dCdt_rates_mean[group][species],
        color=color_species[species],linestyle='--',linewidth=4,alpha=0.6
    )
    
plt.ylim(-1e9,2e9)
#plt.xlim(3e-7,3.1e-7)

plt.savefig('dcdt{}.pdf'.format(groups_str),bbox_inches='tight')


# In[18]:


for group in meta.columns:
    print('\n{}'.format(group))
    for run in meta.loc['Runs',group]:
        print('  run {:>1d}: {:>1.2e} s, {:>4d} steps ({:>1.6e} s/step)'.format(
            run,
            np.sum(progression[group][run].loc[:,'time']),
            len(progression[group][run]),
            np.sum(progression[group][run].loc[:,'time'])/len(progression[group][run]),
        ))


# In[167]:


u1 = 0.99999
rates = [4,4,4,4,1,1,1]
rates.append(np.sum(rates) - np.sum(rates))
print(list(range(len(rates))))
print(rates)
print(np.cumsum(rates))
rxn_idx = np.argwhere(np.cumsum(rates)>=np.sum(rates)*u1)[0][0]
print(np.sum(rates)*u1)
print(rxn_idx)


# In[12]:


group = 'toy4'
V = 1
A = 1e12
T = np.linspace(700,1000,num=50)
K = calc_Eyring(A,meta.loc['Ea Rxn 1 (kcal/mol)',group],T)/calc_Eyring(A,meta.loc['Ea Rxn 2 (kcal/mol)',group],T)
Ca = (-1/4+np.sqrt(V**2/4+4*K*V*200/2)) / (2*K)


# In[13]:


A = 1e12
T = 188
Eas = [1e3,5e3,1e4,5e4,1e5,1e6,1e7,1e8,1e9]
for Ea in Eas:
    print(Ea)
for Ea in Eas:
    print(calc_Ea(A,Ea,T))


# In[35]:


groups = []
#groups += ['toy1','toy2','toy3']
#groups += ['toy4','toy5','toy6']
#groups += ['toy7','toy8','toy9']
#groups += ['toy10','toy11','toy12']
#groups += ['toy13','toy14','toy12']
#groups += ['toy13','toy14']
#groups += ['toy16','toy17','toy18']
#groups += ['toy19','toy20','toy21']
#groups += ['toy22','toy23','toy24','toy25']
groups += ['toy29','toy30','toy26','toy27','toy28']
equilibrium = {
    'AA':{
         'toy1':98.2,  'toy2':97.4,  'toy3':94.8,
         'toy4':98.2,  'toy5':97.4,  'toy6':94.8,
         'toy7':98.2,  'toy8':97.4,  'toy9':94.8,
        'toy10':98.2, 'toy11':97.4, 'toy12':94.8,
        'toy13':   5, 'toy14':  50,
        'toy16':9.5,'toy17':100.0,'toy18':190.5,
        'toy19':9.5,'toy20':100.0,'toy21':190.5
    }
}

meta = pd.read_excel('ToyParameters.xlsx',index_col=0)
meta.drop(columns=[g for g in meta.columns if g not in groups],inplace=True)
for g in groups:
    try:
        meta.loc['Runs',g] = [int(_) for _ in meta.loc['Runs',g].split(',')]
    except:
        meta.loc['Runs',g] = [int(meta.loc['Runs',g])]
#display(meta)


time_ = np.linspace(0,1e-3,num=1000)

for idx,species in enumerate(['A','B','C']):

    # Create figure
    fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='# of {} Molecules'.format(species)
    )
    
    for group in groups:
        X = solve_rateODEs(
                meta,group,time_,
                A=1e12,T=188)
        plt.plot(time_,X[idx],color=color_groups[group],linestyle='-',linewidth=4,alpha=0.8)
        #plt.semilogx(time_,X[idx],color=color_groups[group],linestyle='-',linewidth=2,alpha=0.8)
    plt.plot([0],[meta.loc['Starting {}'.format(species),group]],color='#000000',marker='s',ms=10)
    #plt.xlim(-0.1,0.5)
    
    # Display legend
    handles = [
        mp.Patch(color=color_groups[g],alpha=0.7)
        for g in groups] + [
        Line2D([0],[0],color='#000000',linestyle='-',alpha=0.7,linewidth=4),
        Line2D([0],[0],color='#000000',linestyle='-.',alpha=0.7,linewidth=4)]
    labels = [
        'k1/k3 = {:0.0f}'.format((calc_Eyring(A,meta.loc['Ea Rxn 1 (kcal/mol)',g],T)/calc_Eyring(A,meta.loc['Ea Rxn 3 (kcal/mol)',g],T)))
        for g in groups] + ['hybridMDMC','ODEs']
    #labels = [g for g in groups]
    plt.grid()
    plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.4,1.03))


# In[15]:


A = 1e12
T = 188
Eas = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8]
for Ea in Eas:
    print(Ea)
for Ea in Eas:
    print(calc_Ea(A,Ea,T))