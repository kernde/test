import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
import os
import time
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics, Model, regularizers
from sklearn import preprocessing, feature_extraction, model_selection
from copy import deepcopy
import matplotlib.pyplot as plt
from stellargraph import datasets
from IPython.display import display, HTML
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_loss(history, label, n, loss):
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history[loss],
               color=colors[n], label='Train ' + label)
  plt.semilogy(history.epoch, history.history['val_'+loss],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
matplotlib.use('TkAgg')
import networkx as nx
import pandas as pd
import os
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# rebalance/undersample

rebal_subjects = pd.concat(
    [subjects[subjects == "Fraud"],
    subjects[subjects == "Legit"].sample(subjects[subjects == "Fraud"].shape[0])])

#
train_subjects, val_subjects = model_selection.train_test_split(
    rebal_subjects, train_size=0.5, test_size=None, stratify=rebal_subjects
)

#target_encoding = preprocessing.LabelBinarizer()
#xxx = target_encoding.fit_transform(["Legit", "Fraud"])
train_targets = label_binarize(train_subjects, classes = ['Legit', 'Fraud'])
val_targets = label_binarize(val_subjects, classes = ['Legit', 'Fraud'])

#train_targets = target_encoding.transform(train_subjects) #np.array(train_subjects)
#val_targets = target_encoding.transform(val_subjects)

generator = FullBatchNodeGenerator(G, sparse=True)

train_gen = generator.flow(train_subjects.index, train_targets)

layer_sizes = [16, 16]
gcn = GCN(
    layer_sizes=layer_sizes,
    activations=["elu", "elu"],
    generator=generator,
    dropout=0.3,
    kernel_regularizer=regularizers.l1(2e-4),
)

# Expose the input and output tensors of the GCN model for node prediction, via GCN.in_out_tensors() method:
x_inp, x_out = gcn.in_out_tensors()
# Snap the final estimator layer to x_out
x_out = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)

model = keras.Model(inputs=x_inp, outputs=x_out)

model.compile(
    optimizer=optimizers.Adam(lr=0.01),  # decay=0.001),
    loss=losses.binary_crossentropy,
    metrics=["AUC"],
)

val_gen = generator.flow(val_subjects.index, val_targets)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

cbs = [EarlyStopping(monitor="val_loss", mode="min", patience=4)]
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    verbose=1,
    shuffle=False,
    callbacks=cbs
)

plot_loss(history, "GCN. ", 0, "auc")
plt.legend()
plt.show()

import tikzplotlib
tikzplotlib.save("output_files/aucs_gcn_sup.tex", encoding='utf-8')


### interpretability ####

from stellargraph.interpretability.saliency_maps import IntegratedGradients

graph_nodes = list(G.nodes())
graph_nodes_d = list(G.node_degrees().values())
all_gen = generator.flow(graph_nodes)
preds = model.predict(all_gen)
ftst = preds[:,np.where(subjects == "Fraud"),:] # check for candidate

target_nid = 5

target_idx = graph_nodes.index(target_nid)
target_gen = generator.flow([target_nid])

all_targets = label_binarize(subjects, classes = ['Legit', 'Fraud'])
y_true = all_targets[target_idx]  # true class of the target node
y_pred = preds[0, target_idx]
class_of_interest = np.argmax(y_pred)

print(
    "Selected node id: {}, \nTrue label: {}, \nPredicted scores: {}".format(
        target_nid, y_true, y_pred.round(2)
    )
)

int_grad_saliency = IntegratedGradients(model, train_gen)

integrated_node_importance = int_grad_saliency.get_node_importance(
    target_idx, class_of_interest, steps=30
)

integrated_node_importance.shape

print("\nintegrated_node_importance", integrated_node_importance.round(2))
print("integrate_node_importance.shape = {}".format(integrated_node_importance.shape))
print(
    "integrated self-importance of target node {}: {}".format(
        target_nid, integrated_node_importance[target_idx].round(2)
    )
)
#Check that number of non-zero node importance values is less or equal the number of nodes in target node’s K-hop ego net (where K is the number of GCN layers in the model)

G_ego = nx.ego_graph(G.to_networkx(), target_nid, radius=len(gcn.activations))
print("Number of nodes in the ego graph: {}".format(len(G_ego.nodes())))
print(
    "Number of non-zero elements in integrated_node_importance: {}".format(
        np.count_nonzero(integrated_node_importance)
    )
)

integrate_link_importance = int_grad_saliency.get_integrated_link_masks(
    target_idx, class_of_interest, steps=30
)

#integrate_link_importance_dense = np.array(integrate_link_importance.todense())
#print("integrate_link_importance.shape = {}".format(integrate_link_importance.shape))
#print(
#    "Number of non-zero elements in integrate_link_importance: {}".format(
#        np.count_nonzero(integrate_link_importance.todense())
#    )
#)

#sorted_indices = np.argsort(integrate_link_importance_dense.flatten())
#N = len(graph_nodes)
#integrated_link_importance_rank = [(k // N, k % N) for k in sorted_indices[::-1]]
#topk = 10
# integrate_link_importance = integrate_link_importance_dense
#print(
#    "Top {} most important links by integrated gradients are:\n {}".format(
#        topk, integrated_link_importance_rank[:topk]
#    )
#)

nx.set_node_attributes(G_ego, values={x[0]: {"subject": x[1]} for x in subjects.items()})

integrated_node_importance.max()

integrate_link_importance.max()

node_size_factor = 1e2
link_width_factor = 2

nodes = list(G_ego.nodes())
colors = pd.DataFrame(
    [v[1]["subject"] for v in G_ego.nodes(data=True)], index=nodes, columns=["subject"]
)
colors = np.argmax(target_encoding.transform(colors), axis=1) + 1

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
pos = nx.spring_layout(G_ego)

# Draw ego as large and red
node_sizes = [integrated_node_importance[graph_nodes.index(k)] for k in nodes]
node_shapes = ["o" if w > 0 else "d" for w in node_sizes]

positive_colors, negative_colors = [], []
positive_node_sizes, negative_node_sizes = [], []
positive_nodes, negative_nodes = [], []
node_size_scale = node_size_factor / np.max(node_sizes)
for k in range(len(nodes)):
    if nodes[k] == target_idx:
        continue
    if node_shapes[k] == "o":
        positive_colors.append(colors[k])
        positive_nodes.append(nodes[k])
        positive_node_sizes.append(node_size_scale * node_sizes[k])

    else:
        negative_colors.append(colors[k])
        negative_nodes.append(nodes[k])
        negative_node_sizes.append(node_size_scale * abs(node_sizes[k]))

# Plot the ego network with the node importances
cmap = plt.get_cmap("jet", np.max(colors) - np.min(colors) + 1)
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=positive_nodes,
    node_color=positive_colors,
    cmap=cmap,
    node_size=positive_node_sizes,
    #with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="o",
)
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=negative_nodes,
    node_color=negative_colors,
    cmap=cmap,
    node_size=negative_node_sizes,
    #with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="d",
)
# Draw the target node as a large star colored by its true subject
nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=[target_nid],
    node_size=50 * abs(node_sizes[nodes.index(target_nid)]),
    node_shape="*",
    node_color=[colors[nodes.index(target_nid)]],
    cmap=cmap,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    label="Target",
)

# Draw the edges with the edge importances
edges = G_ego.edges()
weights = [
    integrate_link_importance[graph_nodes.index(u), graph_nodes.index(v)]
    for u, v in edges
]
edge_colors = ["red" if w > 0 else "blue" for w in weights]
weights = link_width_factor * np.abs(weights) / np.max(weights)

ec = nx.draw_networkx_edges(G_ego, pos, edge_color=edge_colors, width=weights)
plt.legend()
plt.colorbar(nc, ticks=np.arange(np.min(colors), np.max(colors) + 1))
plt.axis("off")
plt.show()


import tikzplotlib
tikzplotlib.save("output_files/saliency_map_gcn_full.tex", encoding='utf-8')

