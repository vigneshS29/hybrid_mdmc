#!/usr/bin/env python

#%% Imports
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D
from copy import deepcopy
from sklearn import linear_model
from plotting import formatmpl
from hybrid_mdmc import functions,parsers,mol_classes,classes


#%% Parsing
prefix = 'toy44'
counts,times,selected_rxns = parsers.parse_concentration(prefix+'-1.concentration')
scale,MDMCcycles_scale = parsers.parse_scale(prefix+'-1.scale')
msf = parsers.parse_msf(prefix+'.msf')
rxndf = parsers.parse_rxndf(prefix+'.rxndf')


# %%
column = 'A'
rxndata = rxndf
rxnmatrix = functions.get_rxnmatrix(rxndata,msf)
rxnscaling = scale
progression = functions.get_progression(counts,times,selected_rxns,list(rxndf.keys()),list(msf.keys()))
windowsize_slope = 20
windowsize_scalingpause = 15
windowsize_rxnselection = 15
scalingcriteria_concentration_slope = 0.05
scalingcriteria_concentration_cycles = 15
scalingcriteria_rxnselection_count = 100
PSSrxns = functions.get_PSSrxns(
    rxnmatrix,
    progression[0].loc[50:],
    windowsize_slope,
    windowsize_rxnselection,
    scalingcriteria_concentration_slope,
    scalingcriteria_concentration_cycles,
    scalingcriteria_rxnselection_count,
)
print(PSSrxns)


# %% Formatting
color_map = {_:'#858585' for _ in msf.keys()}
color_map.update({_:'#858585' for _ in rxndf.keys()})
color_map.update({
     'A':'#990000','AA':'#998b00','B':'#000099','C':'#009990'
})
color_map.update({1:'#eb0000',2:'#ebb600',3:'#6e6eff',4:'#6eff96'})


# %%
slopes = np.array([
        linear_model.LinearRegression().fit(
            np.array(range(windowsize_slope)).reshape(-1,1),
            progression[0].loc[:,'A'][idx:idx+windowsize_slope]).coef_[0]
        for idx in range(0,len(progression[0])-windowsize_slope+1)])


# %%
# Create figure
item = 'A'
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='Slope Per Time'
)
plt.plot(
    np.cumsum(progression[0].loc[:len(slopes)-1,'time']),
    slopes,
    color=color_map[item],linewidth=4,alpha=0.7
)
ybounds = 0.05
plt.ylim(-ybounds,ybounds)


# %%
window = 15
rollingmeans = {
    _:[np.mean(progression[0].loc[idx-window:idx,_]) for idx in range(window,len(progression[0]))]
    for _ in [1,2,3,4]
}


# %%
# Create figure
fig,ax,colormap,legendprop = formatmpl(
    figsize=(12,8),
    xstyle='sci',xscilimits=(-2,4),
    ystyle='sci',yscilimits=(-2,4),
    xlabel='time (s)',
    ylabel='Selections'
)
for _ in [1,2,3,4]:
    plt.plot(
        np.cumsum(progression[0].loc[:,'time'][-len(rollingmeans[_]):]),
        np.array(rollingmeans[_])*15,
        color=color_map[_],linewidth=4,alpha=0.5
    )
# Display legend
handles = [mp.Patch(color=color_map[_],alpha=0.7) for _ in [1,2,3,4]]
labels = ['Reaction {}'.format(_) for _ in [1,2,3,4]]
plt.grid()
plt.legend(handles=handles,labels=labels,**legendprop['kwargs'],bbox_to_anchor=(1.4,1.03))


# %%
scalingfactor_adjuster = 0.1
scalingfactor_minimum = 1e-10
display(rxnscaling.loc[8550:,:])
newrxnscaling = functions.scalerxns(
    rxnscaling,
    progression[0].loc[50:],
    PSSrxns,
    windowsize_scalingpause,
    scalingfactor_adjuster,
    scalingfactor_minimum,
    rxnlist='all',
    )
display(newrxnscaling.loc[8550:,:])
# %%
