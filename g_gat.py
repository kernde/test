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

# rebalance/undersample

rebal_subjects = pd.concat(
    [subjects[subjects == "Fraud"],
    subjects[subjects == "Legit"].sample(subjects[subjects == "Fraud"].shape[0])])

#
train_subjects, val_subjects = model_selection.train_test_split(
    rebal_subjects, train_size=0.5, test_size=None, stratify=rebal_subjects
)


target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects) #np.array(train_subjects)
val_targets = target_encoding.transform(val_subjects)

generator = FullBatchNodeGenerator(G, method="gat", k=1) # sparse false needed for saliency maps # , sparse=False

train_gen = generator.flow(train_subjects.index, train_targets)

gat = GAT(
    layer_sizes=[8, train_targets.shape[1]],
    attn_heads=8,
    generator=generator,
    bias=True,
    in_dropout=0.2,
    attn_dropout=0.2,
    activations=["elu", "sigmoid"],
    normalize=None,
    saliency_map_support=True
)

x_inp, predictions = gat.in_out_tensors()

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
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

plot_loss(history, "GAT. ", 0, "auc")
plt.legend()
plt.show()

import tikzplotlib
tikzplotlib.save("output_files/aucs_gat_sup.tex", encoding='utf-8')


# plot embeddings with umap
emb_layer = next(l for l in model.layers if l.name.startswith("graph_attention"))
print(
    "Embedding layer: {}, output shape {}".format(emb_layer.name, emb_layer.output_shape)
)
embedding_model = Model(inputs=x_inp, outputs=emb_layer.output)
all_nodes = subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)
emb = embedding_model.predict(all_gen)
emb.shape
# UMAP embeddings
m = umap.UMAP()
vis_embs = m.fit_transform(holdout_embeddings)

colors = np.linspace(0, 1,2)
colordict = dict(zip(["Fraud", "Legit"], colors))
cols = val_subjects.apply(lambda x: colordict[x])

spfl = np.random.randint(vis_embs.shape[0], size=2000) # sample points for latex (don't accept 10k points in tikz)
plt.scatter(vis_embs[spfl, 0], vis_embs[spfl, 1], alpha=0.1, c = cols.iloc[spfl])
plt.show()




#test_gen = generator.flow(test_subjects.index, test_targets)
#test_metrics = model.evaluate(test_gen)
#print("\nTest Set Metrics:")
#for name, val in zip(model.metrics_names, test_metrics):
#    print("\t{}: {:0.4f}".format(name, val))

all_nodes = subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)

node_predictions = 1-all_predictions.squeeze()

pd.DataFrame({"Predicted": node_predictions, "True": subjects})

emb_layer = next(l for l in model.layers if l.name.startswith("graph_attention"))
print(
    "Embedding layer: {}, output shape {}".format(emb_layer.name, emb_layer.output_shape)
)

embedding_model = Model(inputs=x_inp, outputs=emb_layer.output)

emb = embedding_model.predict(all_gen)
emb.shape #should equal layer size * num attention heads

import umap
m = umap.UMAP()
vis_embs = m.fit_transform(emb.squeeze())
colors = np.linspace(0, 1,2)
colordict = dict(zip(["Fraud", "Legit"], colors))
cols = subjects.apply(lambda x: colordict[x])

spfl = np.random.randint(vis_embs.shape[0], size=10000) # sample points for latex (don't accept 10k points in tikz)
plt.scatter(vis_embs[spfl, 0], vis_embs[spfl, 1], alpha=0.2, c = cols.iloc[spfl])
plt.show()
