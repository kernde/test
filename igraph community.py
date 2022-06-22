import networkx as nx
import igraph as ig
import numpy
import numpy as np
import matplotlib.pyplot as plt

HGnx = HG.to_networkx()
Gnx = nx.Graph(HGnx)  #simplified, remove
print(nx.info(HGnx))
print(nx.info(Gnx))
# Type: MultiGraph
# Number of nodes: 2.842.323
# Number of edges: 2.086.777
# Average degree:  1.4684

print(nx.number_connected_components(Gnx))

# translate the object into igraph
g = ig.Graph.DataFrame(edges)

# degree = g.degree()
#betweenness = g.betweenness() #don't it is slow
degree = np.array([d for n, d in Gnx.degree()])
dg = numpy.unique(degree, return_counts=True)
dg[0]
plt.scatter(dg[0], dg[1])
plt.yscale('log')
from scipy.optimize import curve_fit
def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c
sol, pars = curve_fit(func_powerlaw, xdata = dg[0] , ydata= dg[1] , maxfev=2000, p0 = np.asarray([-1,10**5,0])
plt.plot(dg[0], func_powerlaw(dg[0], *sol), '--')
plt.scatter(dg[0], dg[1])
plt.yscale('log')
plt.xscale('log')

fit = powerlaw.Fit(degree_sequence)

# plt.xscale('log')

#indegree = nx.in_degree_centrality(HGnx) #for directed type.
#outdegree = nx.out_degree_centrality(HGnx)

nx.eigenvector_centrality(Gnx)
nx.katz_centrality(Gnx)

nx.voterank(Gnx)

# centrality = nx.betweenness_centrality(Gnx, k=25) # k << n_nodes, this is a sampled approximation. doc : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
# plot degree versus betweenness
import matplotlib.pyplot as plt

tst = degree.values()
tst2 = np.array(list(tst))* 2842323
plt.plot(tst)
plt.scatter(tst, centrality.values())

# small world property https://en.wikipedia.org/wiki/Small-world_network
nx.sigma(Ge[, 100, 100, 1])
# alternative: nx.omega(HGnx[, 100, nrand, seed])



#

bfs_edges(G, source[, reverse, depth_limit, ...]) # source should a fraud node, until it finds another fraud node.

# import network2tikz