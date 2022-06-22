import stellargraph as sg
from stellargraph.mapper import HinSAGENodeGenerator
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


hinsage_generator = HinSAGENodeGenerator(
    HG, batch_size=1000, num_samples=[5,5], head_node_type="company"
) # inferior or equal time edges only perhaps.

hinsage_model = HinSAGE(
    layer_sizes=[32, 16], activations=["relu","softmax"], generator=hinsage_generator
) # inferior or equal time edges only perhaps.
#hinsage_acc = run_deep_graph_infomax(hinsage_model, hinsage_generator, epochs=epochs)
corrupted_generator = CorruptedGenerator(hinsage_generator)
gen = corrupted_generator.flow(HG.nodes(node_type="company"))
# hinsage_generator.flow(train_subjects.index, train_targets)
# generator.flow(train_subjects.index, train_targets, shuffle=True)
infomax = DeepGraphInfomax(hinsage_model, corrupted_generator)

x_in, x_out = infomax.in_out_tensors()

epochs = 2 # 4 seems enough here
es = EarlyStopping(monitor="loss", min_delta=0, patience=20)

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
history = model.fit(gen, epochs=epochs, verbose=1, callbacks=[es])
from stellargraph.utils import plot_history
plot_history(history)

x_emb_in, x_emb_out = hinsage_model.in_out_tensors()
# for full batch models, squeeze out the batch dim (which is 1)
if hinsage_generator.num_batch_dims() == 2:
    x_emb_out = tf.squeeze(x_emb_out, axis=0)

emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)

y_train = load_frauds_train()
xy_train = pd.merge(pd.DataFrame(nodes_company),y_train, left_index=True, right_index=True, how="left").fillna(0)
y_test = load_frauds_test()
xy_test = pd.merge(pd.DataFrame(nodes_company),y_test, left_index=True, right_index=True, how="left").fillna(0)

nodes_company_gen = hinsage_generator.flow(nodes_company.index) # the index drop is why we keep it
nodes_company_embed = emb_model.predict(nodes_company_gen)

# concat
xy_train = pd.concat([pd.DataFrame(nodes_company_embed), xy_train], axis=1)
xy_test = pd.concat([pd.DataFrame(nodes_company_embed), xy_test], axis=1)

# make a light GBM about it
import lightgbm as ltb
from sklearn.metrics import roc_curve
intr = ['Lat','Lon','TotWorkers']
net = ['pagerank','degree']
embed = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
all = [item for sublist in [intr, net, embed] for item in sublist]

model = ltb.LGBMClassifier(metric='auc', boosting_type='gbdt', objective="binary", num_leaves=31,  # max_depth=8
                           reg_alpha=1, reg_lambda=1, min_data_in_bin=5,
                           n_estimators=100)  # could do hyperoptimisation of parameters as well.
model.fit(xy_train[all], xy_train["Fraud"])
y_hat_sup = model.predict_proba(xy_test[all])[:, 1]
fpr, tpr, threshold = roc_curve(xy_test["Fraud"], y_hat_sup)

plt.plot(fpr,tpr, label = "All, auc:" + str(round(auc(fpr, tpr),2)))
plt.plot([0, 1], [0, 1],'r--')
plt.show()
print(round(auc(fpr, tpr),2))
plt.legend(loc='best'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")

import tikzplotlib
tikzplotlib.save("output_files/ppagerank_perfs_roc.tex", encoding='utf-8')


# plot it
tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(nodes_company_embed[1:4000])

alpha = 0.7
#label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
#node_colours = [label_map[target] for target in node_targets]
matplotlib.use('TkAgg')
plt.figure(figsize=(10, 8))
plt.scatter(
    node_embeddings_2d[:, 0],
    node_embeddings_2d[:, 1],
    c=xy['Fraud'],
    cmap="jet",
    alpha=alpha
)
plt.show()