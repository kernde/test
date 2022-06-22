from stellargraph import StellarGraph, StellarDiGraph
import py2neo
from py2neo import Graph
import pandas as pd
import numpy as np
def dum_sign(dummy_col, threshold=0.005):

    # removes the bind
    dummy_col = dummy_col.copy()

    # what is the ratio of a dummy in whole column
    count = pd.value_counts(dummy_col) / len(dummy_col)

    # cond whether the ratios is higher than the threshold
    mask = dummy_col.isin(count[count > threshold].index)

    # replace the ones which ratio is lower than the threshold by a special name
    dummy_col[~mask] = "others"

    return pd.get_dummies(dummy_col, prefix=dummy_col.name)

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

# Load Nodes
nodes_company = graph.run(
    """
    MATCH (c:Company)
    RETURN ID(c) as ID, c.AddressLat as Lat, c.AddressLon as Lon, 
    toInteger(c.WorkersDirectorsNumber) + toInteger(c.WorkersEmployeesNumber) + toInteger(c.WorkersServantsNumber) + 
    toInteger(c.WorkersWorkersNumber) as TotWorkers, toInteger(c.community) as Community, toInteger(c.degree) as Degree
    """
).to_data_frame().set_index("ID")
nodes_company = nodes_company.fillna(0)
nodes_company = nodes_company.apply(pd.to_numeric)
nodes_company = pd.concat([nodes_company.drop('Community', axis=1), dum_sign(nodes_company['Community'])], axis=1)
nodes_company = (nodes_company-nodes_company.min())/(nodes_company.max()-nodes_company.min()) # normalize for stellargraph
nodes_company

nodes_persons = graph.run(
    """
    MATCH (p:Person)
    RETURN ID(p) as ID, 1 as Dummy
    """
).to_data_frame().set_index("ID")

# load edges
edges = graph.run(
    """
    MATCH (s) -[:OWNS|:WORKS]-> (t)
    RETURN ID(s) AS source, ID(t) AS target
    """
).to_data_frame()

# we could delete duplicate relationships (e.g. both P & A roles account for two relationships)
edges = edges.drop_duplicates()

HG = StellarDiGraph({"company" : nodes_company, "person" : nodes_persons}, edges)

# load nodes to make labels Y for later tasks
labels_fraud = graph.run(
    """
    MATCH (c:Company)-[:Involved_in]-> (f:Fraud)
    RETURN ID(c) as ID, c.CompanyId as CompanyId, count(f) as y
    """
).to_data_frame().set_index("ID")

labels_fraud = graph.run(
    """
    MATCH (p:Person)-[:Involved_in]-> (f:Fraud)
    RETURN ID(f) as ID
    """
).to_data_frame().set_index("ID")
# labels_fraud.y.hist() # inspect proportions


