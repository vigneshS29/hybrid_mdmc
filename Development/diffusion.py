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
