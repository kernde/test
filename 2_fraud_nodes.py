from py2neo import Graph
import pandas as pd
import networkx as nx
import numpy as np

def load_hin(keep_isolated_nodes=False):
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    companies = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH (s:Company)
        RETURN ID(s) as ID
        """
    ).to_data_frame()

    admins = graph.run( # we need to keep the unlinked ones later
        """
        MATCH (p:Person)
        RETURN ID(p) as ID
        """
    ).to_data_frame()

    fraud = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH ()-[:Involved_in]-> (f:Fraud)
        where f.Registration_date < '20180101'
        RETURN ID(f) as ID, CASE f.FileType WHEN 'RR12' THEN 0.1 ELSE 1 END as weight
        """
    ).to_data_frame().set_index('ID') # say RR12 is 0.1 (minor misconduct), the rest is 1 (sever cases).

    p_pagerank = fraud.to_dict()['weight'] # personalized scores for personalized pagerank.

    edges = graph.run(
        """
        MATCH (s) -[]-> (t)
        RETURN ID(s) AS source, ID(t) AS target
        """
    ).to_data_frame()
    edges = edges.drop_duplicates()

    H = nx.from_pandas_edgelist(edges, 'source', 'target')

    if keep_isolated_nodes:  # if we don't do that beware of bias in comparing the two alternatives.
        singleton_c = companies[(~companies['ID'].isin(edges.source)) & (~companies['ID'].isin(edges.target))]
        H.add_nodes_from(singleton_c.ID)
        singleton_a = admins[(~admins['ID'].isin(edges.source)) & (~admins['ID'].isin(edges.target))]
        H.add_nodes_from(singleton_a.ID)
    H = H.to_undirected()
    return H, p_pagerank

def load_fraud_test():
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    fraud = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH (c)-[:Involved_in]-> (f:Fraud)
        where f.Registration_date > '20180101'
        and f.FileType <> 'RR12'
        RETURN ID(c) as ID, 1 as Fraud
        """
    ).to_data_frame().set_index('ID') # say RR12 is 0.1 (minor misconduct), the rest is 1 (sever cases).
    return fraud

def load_neverseenbefore_fraud_test():
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    fraud_known = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH (c)-[:Involved_in]-> (f:Fraud)
        where f.Registration_date < '20180101'
        RETURN ID(c) as ID, 1 as Fraud
        """
    ).to_data_frame().set_index('ID')

    fraud_new = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH (c)-[:Involved_in]-> (f:Fraud)
        where f.Registration_date > '20180101' and f.Registration_date < '20220606'
        RETURN ID(c) as ID, 1 as Fraud
        """
    ).to_data_frame().set_index('ID')

    fraud = fraud_new[~fraud_new.index.isin(fraud_known.index)]

    return fraud

def load_company_ID():
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    cid = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH (c:Company)
        RETURN ID(c) as ID
        """
    ).to_data_frame() # say RR12 is 0.1 (minor misconduct), the rest is 1 (sever cases).
    return cid

def load_admin_ID():
    # extended stands for additional edges from common employments, hence a more connected company graph.
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

    aid = graph.run(  # we need to keep the unlinked ones later
        """
        MATCH (p:Person)
        RETURN ID(p) as ID
        """
    ).to_data_frame() # say RR12 is 0.1 (minor misconduct), the rest is 1 (sever cases).
    return aid

