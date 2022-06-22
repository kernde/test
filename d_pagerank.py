import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", "fraud"))

H, pprank = load_hin(keep_isolated_nodes=True)

# nodes = list(deg.keys())
pr = nx.pagerank(H, personalization=pprank, alpha=0.85, max_iter=100, tol=1e-06)

predictions = [
    {"ID": nodes, "pagerank": pr}
    for nodes, pr in zip(pr.keys(), pr.values())
]

tst = pr.values()
plt.hist(pr.values(), log=True)
plt.xscale("log") # log looks weird

# evaluate on a test "out of time".
frauds = load_fraud_test() #load_neverseenbefore_fraud_test
x = pd.DataFrame.from_dict(pr, orient='index', columns=["prediction"])

xy = pd.merge(x, frauds, left_index=True, right_index=True, how='left')
xy = xy.fillna(0)

# to inspect performance per node type.
cid = np.array(load_company_ID()['ID'])
pid = np.array(load_admin_ID()['ID'])

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(xy['Fraud'], xy['prediction'], drop_intermediate=True)
plt.plot(fpr,tpr, label="Global")
fpr, tpr, threshold = roc_curve(xy.loc[cid]['Fraud'], xy.loc[cid]['prediction'], drop_intermediate=True)
plt.plot(fpr,tpr, label="Companies")
fpr, tpr, threshold = roc_curve(xy.loc[pid]['Fraud'], xy.loc[pid]['prediction'], drop_intermediate=True)
plt.plot(fpr,tpr, label="Individuals")
plt.plot([0, 1], [0, 1],'b--')
plt.legend(loc='best'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")

import tikzplotlib
tikzplotlib.save("output_files/ppagerank_perfs_roc.tex", encoding='utf-8')

# second check: new fraudsters


# save it back to graph.
graph.evaluate(
    """
    UNWIND $predictions AS prediction
    MATCH (c:Company)
    Where ID(c) = prediction.ID
    SET c.pagerank = prediction.pagerank
    """,
    {"predictions": predictions},
)
graph.evaluate(
    """
    UNWIND $predictions AS prediction
    MATCH (p:Person)
    Where ID(p) = prediction.ID
    SET p.pagerank = prediction.pagerank
    """,
    {"predictions": predictions},
)

# check results
graph.run(
    "MATCH (c:Company) RETURN ID(c), c.pagerank limit 50"
).to_data_frame() # degree of both direct companies and common employees !