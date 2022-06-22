import matplotlib
matplotlib.use('Agg')
from stellargraph import StellarDiGraph
from py2neo import Graph
import pandas as pd

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

# Load Nodes
nodes_company = graph.run(
    """
    MATCH (c:Company)
    RETURN ID(c) as ID, c.AddressLat as Lat, c.AddressLon as Lon, 
    toInteger(c.WorkersDirectorsNumber) + toInteger(c.WorkersEmployeesNumber) + toInteger(c.WorkersServantsNumber) + 
    toInteger(c.WorkersWorkersNumber) as TotWorkers, c.pagerank as pagerank, toInteger(c.degree) as degree 
    """
).to_data_frame().set_index("ID")
nodes_company = nodes_company.fillna(0)
nodes_company = nodes_company.apply(pd.to_numeric)
nodes_company = (nodes_company-nodes_company.min())/(nodes_company.max()-nodes_company.min()) # normalize for stellargraph
nodes_company

nodes_persons = graph.run(
    """
    MATCH (p:Person)
    RETURN ID(p) as ID, 
    case p.Gender
    when 'M' then 1
    when 'V' then 0 end as gender,
    p.pagerank as pagerank
    """
).to_data_frame().set_index("ID")
nodes_persons = nodes_persons.fillna(0) # singleton have 0 pagerank
nodes_persons = (nodes_persons-nodes_persons.min())/(nodes_persons.max()-nodes_persons.min()) # normalize for stellargraph
nodes_persons
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


def load_frauds_train():
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    fraud = graph.run(
        """
        MATCH (c)-[:Involved_in]-> (f:Fraud)
        where f.Registration_date > '20180101'
        and f.Registration_date < '20190101'
        and f.FileType <> 'RR12'
        RETURN ID(c) as ID, 1 as Fraud
        """
    ).to_data_frame().set_index('ID')

    return fraud

def load_frauds_test():
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    fraud = graph.run(
        """
        MATCH (c)-[:Involved_in]-> (f:Fraud)
        where f.Registration_date > '20190101'
        and f.FileType <> 'RR12'
        RETURN ID(c) as ID, 1 as Fraud
        """
    ).to_data_frame().set_index('ID')

    return fraud