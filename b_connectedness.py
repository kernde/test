from py2neo import Graph
import pandas as pd
import networkx as nx, matplotlib.pyplot as plt
from .graphs_loading import load_company_graph
import numpy.ma as ma

G = load_company_graph(extended=False,output_type='NetworkX', keep_isolated_nodes= True)
Ge = load_company_graph(extended=True,output_type='NetworkX', keep_isolated_nodes= True)

print(nx.number_connected_components(G))
print(list(nx.connected_components(G)))
component_map = { }
components = nx.connected_components(G)
for nodes in components:
    print(len(nodes))
    break

sizes_subgraphs_G = [len(c) for c in nx.connected_components(G)]
plt.hist(sizes_subgraphs_G)
unique, counts = np.unique(sizes_subgraphs_G, return_counts=True)
plt.bar(unique, counts, log=True, ec="k", align="edge")
plt.xscale("log")


sizes_subgraphs_G = [len(c) for c in nx.connected_components(G)]
sizes_subgraphs_Ge = [len(c) for c in nx.connected_components(Ge)]

unique, counts = np.unique(sizes_subgraphs_G, return_counts=True)
uniquee, countse = np.unique(sizes_subgraphs_Ge, return_counts=True)

plt.bar(unique, np.maximum(np.log(counts),0.1), ec="k", align="edge", alpha=0.5, label="Company")
plt.bar(uniquee, np.maximum(np.log(countse),0.1), ec="k", align="edge", alpha=0.5, label="Company Dense")
plt.xlim((0,140))
plt.ylim((-0.2,14))
plt.xlabel("Connected Subgraph Size")
plt.ylabel("Log Number of Subgraphs")
plt.legend(loc="upper right")

import tikzplotlib
tikzplotlib.save("output_files/connected_components.tex")

print("Average Size: " + str(np.mean(sizes_subgraphs_G)))
print("Average Size (dense): " + str(np.mean(sizes_subgraphs_Ge)))

print("Large Component Size: " + str(np.max(unique)))
print("Large Component Size (dense): " + str(np.max(uniquee)))
