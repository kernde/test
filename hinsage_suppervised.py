import stellargraph as sg
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras import Model, optimizers, losses, metrics
from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
import keras
import multiprocessing
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
# from IPython.display import display, HTML
import matplotlib
matplotlib.use('Agg')

hinsage_generator = HinSAGENodeGenerator(
    HG, batch_size=1000, num_samples=[3,3], head_node_type="company"
) # inferior or equal time edges only perhaps.

train_subjects, test_subjects = model_selection.train_test_split(
    nodes_company, train_size=0.5, test_size=None, #stratify=claims_nodes
)
#test_gen = hinsage_generator.flow(test_subjects.index)
#train_gen = hinsage_generator.flow(train_subjects.index)

y = pd.merge(nodes_company, labels_fraud['y'].clip(upper=1), how="left", left_index=True, right_index=True).fillna(0)['y']
train_targets = y.loc[train_gen.ids]
test_targets = y.loc[test_gen.ids]

train_gen = hinsage_generator.flow(train_subjects.index, y.loc[train_subjects.index])
test_gen = hinsage_generator.flow(test_subjects.index, y.loc[test_subjects.index])

hinsage_model = HinSAGE(
    layer_sizes=[16, 16], activations=["relu","softmax"], generator=hinsage_generator
)

x_inp, x_out = hinsage_model.in_out_tensors()
prediction = layers.Dense(units=1, activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
metrics = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='auc'),
]

model.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.nn.sigmoid_cross_entropy_with_logits,
    metrics=metrics
)

train_gen = hinsage_generator.flow(train_subjects.index, train_targets, shuffle=True)
test_gen = hinsage_generator.flow(test_subjects.index, test_targets)

history = model.fit(
    train_gen, epochs=2, validation_data=test_gen, verbose=1, shuffle=False
)
