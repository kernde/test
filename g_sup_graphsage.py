import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# check this for more details on method: https://antonsruberts.github.io/graph/graphsage/

train_subjects, test_subjects = model_selection.train_test_split(
    subjects, train_size=0.5, test_size=None, stratify=subjects
)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)

batch_size = 1000
num_samples = [5, 5]

generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)

graphsage_model = GraphSAGE(
    layer_sizes=[16, 16], generator=generator, bias=True, dropout=0.5,
)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.0005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

test_gen = generator.flow(test_subjects.index, test_targets)
history = model.fit(
    train_gen, epochs=2, validation_data=test_gen, verbose=1, shuffle=False
)

sg.utils.plot_history(history)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

all_nodes = subjects.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict(all_mapper)


node_predictions = target_encoding.inverse_transform(all_predictions)

df = pd.DataFrame({"Predicted": node_predictions, "True": subjects})
df.head(10)
