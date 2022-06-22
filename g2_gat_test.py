import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
import os
import time
import sys
import stellargraph as sg
from copy import deepcopy
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT, GraphAttention

from tensorflow.keras import layers, optimizers, losses, metrics, models, Model
from sklearn import preprocessing, feature_extraction, model_selection
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from stellargraph import datasets
from IPython.display import display, HTML

dataset = datasets.Cora()
display(HTML(dataset.description))
G, subjects = dataset.load()
print(G.info())
train_subjects, test_subjects = model_selection.train_test_split(
    subjects, train_size=140, test_size=None, stratify=subjects
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=500, test_size=None, stratify=test_subjects
)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)

all_targets = target_encoding.transform(subjects)

generator = FullBatchNodeGenerator(G, method="gat", sparse=False)

train_gen = generator.flow(train_subjects.index, train_targets)

gat = GAT(
    layer_sizes=[8, train_targets.shape[1]],
    attn_heads=8,
    generator=generator,
    bias=True,
    in_dropout=0,
    attn_dropout=0,
    activations=["elu", "softmax"],
    normalize=None,
    saliency_map_support=True,
)

x_inp, predictions = gat.in_out_tensors()

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    weighted_metrics=["acc"],
)

val_gen = generator.flow(val_subjects.index, val_targets)

N = G.number_of_nodes()
history = model.fit(
    train_gen, validation_data=val_gen, shuffle=False, epochs=10, verbose=1
)
sg.utils.plot_history(history)
test_gen = generator.flow(test_subjects.index, test_targets)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))