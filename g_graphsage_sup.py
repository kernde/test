import networkx as nx
import pandas as pd
import os
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.layer.graphsage import MeanPoolingAggregator, AttentionalAggregator, MeanAggregator
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import umap

# rebalance/undersample
rebal_subjects = pd.concat(
    [subjects[subjects == "Fraud"],
    subjects[subjects == "Legit"].sample(subjects[subjects == "Fraud"].shape[0])])

train_subjects, val_subjects = model_selection.train_test_split(
    rebal_subjects, train_size=0.5, test_size=None, stratify=rebal_subjects
)

#val_subjects, test_subjects = model_selection.train_test_split(
#    test_subjects, train_size=0.2, test_size=None, stratify=test_subjects
#)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects) #np.array(train_subjects)
val_targets = target_encoding.transform(val_subjects)
#test_targets = target_encoding.transform(test_subjects)

# graphSage
batch_size = 1000
num_samples = [10, 5]

generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)

def GSage(aggregator=MeanAggregator):
    # aggregator: [MeanPoolingAggregator, MeanAggregator, AttentionalAggregator]
    graphsage_model = GraphSAGE(
        layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5, aggregator=aggregator
    )
    x_inp, x_out = graphsage_model.in_out_tensors()
    prediction = layers.Dense(1, activation="sigmoid")(x_out)
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.binary_crossentropy,
        metrics=['AUC'],
    )
    test_gen = generator.flow(val_subjects.index, val_targets)
    cbs = [EarlyStopping(monitor="val_loss", mode="min", patience=3)]
    history = model.fit(
        train_gen, epochs=10, validation_data=test_gen, verbose=1, shuffle=False, callbacks=cbs
    )
    return history, model
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

history_ma, model_ma = GSage()
history_mpa, model_mpa = GSage(aggregator=MeanPoolingAggregator)
history_aa, model_aa = GSage(aggregator=AttentionalAggregator)

plot_loss(history_ma, "Mean.Agg. ", 0, "auc")
plot_loss(history_mpa, "Mean.P.Agg ", 1, "auc")
plot_loss(history_aa, "Attent.Agg ", 2, "auc")
plt.legend()
#matplotlib.use('TkAgg') # depends on pycharm mode
plt.show()

import tikzplotlib
tikzplotlib.save("output_files/aucs_graphsage_sup.tex", encoding='utf-8')


embedding_model = Model(inputs=[model_aa.layers[2].input, model_aa.layers[0].input, model_aa.layers[1].input],
                        outputs=[model_aa.layers[-3].output]) # output grabs the last reshape layer with shape (None, 32)

# make prediction
holdout_nodes = val_subjects.index
holdout_labels = val_targets

holdout_generator = generator.flow(holdout_nodes)

holdout_embeddings = embedding_model.predict(holdout_generator)
# UMAP embeddings

m = umap.UMAP()

vis_embs = m.fit_transform(holdout_embeddings)


colors = np.linspace(0, 1,2)
colordict = dict(zip(["Fraud", "Legit"], colors))
cols = val_subjects.apply(lambda x: colordict[x])

spfl = np.random.randint(vis_embs.shape[0], size=2000) # sample points for latex (don't accept 10k points in tikz)
plt.scatter(vis_embs[spfl, 0], vis_embs[spfl, 1], alpha=0.1, c = cols.iloc[spfl])
plt.show()

import tikzplotlib
tikzplotlib.save("output_files/umap_embed_aa.tex", encoding='utf-8')

