from py2neo import Graph
import pandas as pd
import networkx as nx
from .graphs_loading import load_company_graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

Ge = load_company_graph(extended=True,output_type='NetworkX')

deg = dict(Ge.degree())

nodes = list(deg.keys())
degree = list(deg.values())

predictions = [
    {"ID": nodes, "degree": degree}
    for nodes, degree in zip(nodes, degree)
]

graph.evaluate(
    """
    UNWIND $predictions AS prediction
    MATCH (c:Company)
    Where ID(c) = prediction.ID
    SET c.degree = prediction.degree
    """,
    {"predictions": predictions},
)

# check results
graph.run(
    "MATCH (c:Company) RETURN ID(c), c.degree limit 50"
).to_data_frame() # degree of both direct companies and common employees !

