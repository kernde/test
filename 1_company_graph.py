from py2neo import Graph
import pandas as pd
import networkx as nx
import numpy as np

def load_company_graph(extended=False, output_type='NetworkX', keep_isolated_nodes= True):

    # extended stands for additional edges from common employments, hence a more connected company graph.

    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    companies = graph.run( # we need to keep the unlinked ones later
        """
        MATCH (s:Company)
        RETURN ID(s) as ID
        """
    ).to_data_frame()

    # load edges
    edges = graph.run(
        """
        MATCH (s:Company) -[:OWNS]-> (t:Company)
        RETURN ID(s) AS source, ID(t) AS target
        """
    ).to_data_frame()

    if extended:
        edges_w = graph.run(  # OPTIONAL MATCH (s:Company) -[:OWNS]-> (t:Company)
            """
            MATCH  (s:Company) <- [:WORKS]- (w:Person) -[:WORKS]-> (t:Company)
            RETURN ID(s) AS source, ID(t) AS target
            """
        ).to_data_frame()
        edges = pd.concat([edges_w, edges])

    edges = edges.drop_duplicates()

    if output_type == 'NetworkX':
        G = nx.from_pandas_edgelist(edges, 'source', 'target')
        if keep_isolated_nodes: #if we don't do that beware of bias in comparing the two alternatives.
            singleton = companies[(~companies['ID'].isin(edges.source)) & (~companies['ID'].isin(edges.target))]
            G.add_nodes_from(singleton.ID)
        G = G.to_undirected()
        return G
    if output_type == 'PandasEdges':
        return edges