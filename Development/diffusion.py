#!/usr/bin/env python
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

import numpy as np
from collections import defaultdict

# ToDo
#   - catch error where there is no path between two voxels (may already be handled)
#   - documentation
#   - parallelize; get_diffusionratematrix_from_transitionratematrix performs MANY
#     independent calculations. Parallel handling would provide tremendous speedup.


class DiffusionGraph():
    def __init__(self, edges=defaultdict(list), weights={}):
        self.edges = edges
        self.weights = weights

    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight


def get_timegraph_from_ratematrix(ratematrix):
    timegraph = DiffusionGraph(
        edges={
            idx: [
                col for col, _ in enumerate(ratematrix[idx, :]) if _]
            for idx in range(len(ratematrix))
        },
        weights={
            (ridx, cidx): 1/val
            for ridx, row in enumerate(ratematrix) for cidx, val in enumerate(row) if val
        })
    return timegraph


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]
        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)
                                   ] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        next_destinations = {
            node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return None
            # next node is the destination with the lowest weight
        current_node = min(next_destinations,
                           key=lambda k: next_destinations[k][1])
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


def get_diffusiontime_with_dijkstra(time_graph, initial, end):
    path = dijsktra(time_graph, initial, end)
    if not path:
        return [1e99]
    return [time_graph.weights[(path[idx], path[idx+1])] for idx in range(len(path)-1)]


def get_diffusionratematrix_from_transitionratematrix(transitionrate_matrix):
    transitiontime_graph = get_timegraph_from_ratematrix(transitionrate_matrix)
    diffusionratematrix = np.zeros(
        (len(transitiontime_graph.edges), len(transitiontime_graph.edges)))
    for ridx in range(len(diffusionratematrix)):
        for cidx in range(len(diffusionratematrix)):
            if ridx == cidx:
                diffusionratematrix[ridx, cidx] = 1e99
                continue
            diffusionratematrix[ridx, cidx] = 1/np.sum(
                get_diffusiontime_with_dijkstra(transitiontime_graph, ridx, cidx))
    return diffusionratematrix


def calc_neighbor_diffusionrate(traj_file,datafile_molecules,voxels,msf):

    # Create a voxelID2idx map
    # This could be passed in as an argument, but it is cheap to recreate and simplifies things to recreate it as needed
    # Creating a voxel class would simplify the whole algorithm; this map could be included in the class
    voxelID2idx = {k:idx for idx,k in enumerate(sorted(list(voxels.keys())))}

    # Create a dictionary to hold the voxel number of each molecule for each frame.
    # The entries are a list of voxels, indexed to datafile_molecules.ids
    voxels_byframe = {}

    # Loop over all of the requested frames in the trajectory
    for frame_atoms,timestep,box in frame_generator(
            traj_file,start=0,end=-1,every=1,unwrap=True,adj_list=datafile_molecules.atom_ids,return_prop=False):

        # Create a temporary MoleculeList instance to hold the molecule information
        molecules_thisframe = MoleculeList(
            ids=datafile_molecules.ids,
            mol_types=datafile_molecules.mol_types,
            atom_ids=datafile_molecules.atom_ids
        )

        # Calculate the COG and voxel of each molecule
        molecules_thisframe.calc_cog(frame_atoms)
        molecules_thisframe.get_voxels(voxels)

        # Add the timestep: voxel_list entry to the voxels_byframe dictionary for this frame
        voxels_byframe[timestep] = molecules_thisframe.voxels

    # Create the dictionary that will hold an array for each species time
    # Each array will eventually hold the diffusion rate for the species traveling From voxel "i" TO voxel "j"
    # in element row i, column j.
    # The diffusion "rate" is (number of occurences) / (time)
    neighbor_diffusionrates = {species:np.zeros(len(voxels),len(voxels)) for species in msf.keys()}
    
    # Create a sorted list of the timesteps
    timesteps = sorted(voxels_byframe.keys())

    # Loop over each molecule provided in the datafile_molecule object
    for midx,moleculeID in enumerate(datafile_molecules.ids):

        # Loop through every timestep in the provided trajectory
        for idx in range(1,len(timesteps)):

            # Add the voxel change to the approriate array that this molecule experienced for this step in the trajectory
            neighbor_diffusionrates[ datafile_molecules.mol_types[midx] ][ voxelID2idx[voxels_byframe[timesteps[idx-1]][midx]], voxelID2idx[voxels_byframe[timesteps[idx]][midx]] ] += 1

    # To go from a count of the number of occurances to the "rate," divide by the total time
    for k,v in neighbor_diffusionrates.items():
        neighbor_diffusionrates[k] /=  (timesteps[-1] - timesteps[0])

    # Get the direct diffusion rate for each i-to-j voxel transition using Dijsktra's algorithm
    
