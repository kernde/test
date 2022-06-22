import igraph as ig
import leidenalg as la

from cdlib import algorithms

import networkx as nx
G = nx.karate_club_graph()
coms = algorithms.leiden(G)

# clustering only on the largest component