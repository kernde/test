import matplotlib
matplotlib.use('Agg')
from stellargraph import StellarDiGraph, StellarGraph
from py2neo import Graph
import pandas as pd

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

G = load_company_graph(extended=True, output_type='NetworkX', keep_isolated_nodes= True)
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
largest_comp = np.array(G0.nodes)

companies = graph.run(  # we need to keep the unlinked ones later
    """
    MATCH (c:Company)
    RETURN ID(c) as ID, c.AddressLat as Lat, c.AddressLon as Lon, 
    toInteger(c.WorkersDirectorsNumber) as DirectNumber, 
    toInteger(c.WorkersEmployeesNumber) as EmpNumber,
    toInteger(c.WorkersServantsNumber) as ServNumber,
    toInteger(c.WorkersWorkersNumber) as WorkerNumber,
    toInteger(c.WorkersDirectorsNumber) + toInteger(c.WorkersEmployeesNumber) + toInteger(c.WorkersServantsNumber) + 
    toInteger(c.WorkersWorkersNumber) as TotWorkers, toInteger(c.EstablishmentYear) as establishyear, c.degree as degree
    """
).to_data_frame().set_index("ID")
companies = companies.loc[largest_comp]
companies = companies.fillna(0) # care with the meaning of this imputation (depends on the features).
companies = companies.apply(pd.to_numeric)
companies = (companies-companies.min())/(companies.max()-companies.min()) # normalize for stellargraph

edges = graph.run(
    """
    MATCH (s:Company) -[:OWNS]-> (t:Company)
    RETURN ID(s) AS source, ID(t) AS target
    """
).to_data_frame()
edges_w = graph.run(  # OPTIONAL MATCH (s:Company) -[:OWNS]-> (t:Company)
    """
    MATCH  (s:Company) <- [:WORKS]- (w:Person) -[:WORKS]-> (t:Company)
    RETURN ID(s) AS source, ID(t) AS target
    """
).to_data_frame()
edges = pd.concat([edges_w, edges])
edges = pd.concat([edges.min(axis=1),edges.max(axis=1)], axis =1, keys=['source', 'target'])
edges = edges.drop_duplicates(ignore_index=True)
edges = edges[edges.source.isin(largest_comp)]
edges = edges[edges.target.isin(largest_comp)]

G = StellarGraph(nodes=companies, edges=edges)

## fraud label
fraudsc = graph.run( # if we want also admin involved # Optional MATCH (c:Company)<-[:WORKS]- (w:Person)-[:Involved_in]->(f:Fraud)
        """
        MATCH (c:Company)-[:Involved_in]->(f:Fraud)
        where f.FileType <> 'RR12'
        RETURN ID(c) as ID, 'Fraud' as Fraud
        """
    ).to_data_frame().set_index('ID') # and f.Registration_date < '20200601'
fraudsa = graph.run( # if we want also admin involved # Optional MATCH (c:Company)<-[:WORKS]- (w:Person)-[:Involved_in]->(f:Fraud)
        """
        MATCH (c:Company)<-[:WORKS]- (w:Person)-[:Involved_in]->(f:Fraud)
        where f.FileType <> 'RR12'
        RETURN ID(c) as ID, 'Fraud' as Fraud
        """
    ).to_data_frame().set_index('ID') # and f.Registration_date < '20200601'
frauds = pd.concat([fraudsc, fraudsa])
frauds = frauds[~frauds.index.duplicated(keep='first')]

subjects = pd.merge(companies, frauds, left_index=True, right_index=True, how="left").fillna('Legit')["Fraud"]

from collections import Counter
Counter(subjects)
