from stellargraph import StellarGraph
import py2neo
from py2neo import Graph
import pandas as pd

from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph, StellarDiGraph
import py2neo
from py2neo import Graph
import pandas as pd
import numpy as np

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

# Load Nodes
nodes_company = graph.run(
    """
    MATCH (c:Company)
    RETURN ID(c) as ID
    """
).to_data_frame().set_index("ID")

nodes_persons = graph.run(
    """
    MATCH (p:Person)
    RETURN ID(p) as ID
    """
).to_data_frame().set_index("ID")

nodes_fraud = graph.run(
    """
    MATCH (f:Fraud)<-[:Involved_in]-(x)
    RETURN ID(f) as ID
    """
).to_data_frame().set_index("ID") # to avoid the singleton
nodes_fraud = nodes_fraud[~nodes_fraud.index.duplicated(keep='first')]

# load edges
edges = graph.run(
    """
    MATCH (s) -[:OWNS| :WORKS| :Involved_in]-> (t)
    RETURN ID(s) AS source, ID(t) AS target
    """
).to_data_frame()

# we could delete duplicate relationships (e.g. both P & A roles account for two relationships)
edges = edges.drop_duplicates()

HG = StellarGraph({"company" : nodes_company, "person" : nodes_persons, "fraud" : nodes_fraud}, edges)

# Create the random walker
rw = BiasedRandomWalk(HG)
# specify the metapath schemas as a list of lists of node types

walks = rw.run(
    nodes=list(HG.nodes("fraud")),  # root nodes
    n=5,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2,
    length=5,  # maximum length of a random walk
)

from collections import Iterable
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item

from collections import Counter
fraud_rw_score = Counter(flatten(walks))
max_visit = max(fraud_rw_score.values())
for k in fraud_rw_score:
  fraud_rw_score[k] = fraud_rw_score[k]/max_visit

nodes = list(fraud_rw_score.keys())
fraud_rw = list(fraud_rw_score.values())
#predicted_class = np.array(["Y", "X"])

predictions = [
    {"ID": nodes.item(), "rsr_proximity": fraud_rw}
    for nodes, fraud_rw in zip(nodes, fraud_rw)
] #.item() because it is stored as numpy int, not compatible with cypher.

graph.evaluate( #could also write to individuals.
    """
    UNWIND $predictions AS prediction
    MATCH (c:Company)
    Where ID(c) = prediction.ID
    SET c.rsr_proximity = prediction.rsr_proximity
    """,
    {"predictions": predictions},
)

