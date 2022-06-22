from py2neo import Graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

# load edges
edges_w = graph.run( #OPTIONAL MATCH (s:Company) -[:OWNS]-> (t:Company)
    """
    MATCH  (s:Company) <- [:WORKS]- (w:Person) -[:WORKS]-> (t:Company)
    RETURN ID(s) AS source, ID(t) AS target
    """
).to_data_frame()

edges_c = graph.run(
    """
    MATCH (s:Company) -[:OWNS]-> (t:Company)
    RETURN ID(s) AS source, ID(t) AS target
    """
).to_data_frame()

edges = pd.concat([edges_w, edges_c]).drop_duplicates()

G = nx.from_pandas_edgelist(edges, 'source', 'target')
G = G.to_undirected() # the "to_undirected()" merges bidirectional relations if any remain

partition = community_louvain.best_partition(G) # ~3-4 minutes

#keep the top and leave miscelaneous catagorie
# plt.bar(partition.keys(), partition.values(), color='g') #very slow

#test = partition.values()
#from collections import Counter
#va = Counter(partition.values()).most_common(500) # say top 500 company (arbitrary)

# counts = np.bincount(list(partition.values()))
# counts[np.where(counts > 100)] # all communities with at least 100 companies in there.

import numpy as np

nodes = list(partition.keys())
community = list(partition.values())
#predicted_class = np.array(["Y", "X"])

predictions = [
    {"ID": nodes, "community": community}
    for nodes, community in zip(nodes, community)
]

graph.evaluate(
    """
    UNWIND $predictions AS prediction
    MATCH (c:Company)
    Where ID(c) = prediction.ID
    SET c.community = prediction.community
    """,
    {"predictions": predictions},
)

# check results
graph.run(
    "MATCH (c:Company) RETURN ID(c), c.community limit 50"
).to_data_frame() # some NaN values for disconnected nodes.

