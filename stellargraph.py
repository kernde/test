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
import multiprocessing
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
# from IPython.display import display, HTML
import matplotlib
matplotlib.use('Agg')

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

train_subjects, test_subjects = model_selection.train_test_split(
    nodes_company, train_size=0.5, test_size=None, #stratify=claims_nodes
)

test_gen = hinsage_generator.flow(test_subjects.index)
train_gen = hinsage_generator.flow(train_subjects.index)

test_embeddings = emb_model.predict(test_gen)
train_embeddings = emb_model.predict(train_gen)

y = pd.merge(nodes_company, labels_fraud['y'].clip(upper=1), how="left", left_index=True, right_index=True).fillna(0)['y']

# Option 1: We have kept the fraud labels in graph (how does it influence embeddings ?)
# some generators yield predictions in a different order to the .flow argument,
# so we need to get everything lined up correctly
#reorder = lambda sequence, subjects: subjects
#ordered_test_subjects = reorder(test_gen, y) # lambda sequence wtf ?
#ordered_train_subjects = reorder(train_gen, y) # y.loc[test_gen.ids] # simpler

### replace by a boosting this crap.
lr = LogisticRegression()
lr.fit(train_embeddings, y.loc[train_gen.ids])

y_pred = lr.predict_proba(test_embeddings)
#acc = (y_pred == ordered_test_subjects).mean()

# comp to intrinsic features:
lr = LogisticRegression()
lr.fit(train_subjects, y.loc[train_gen.ids])
y_pred = lr.predict_proba(test_subjects)
import lightgbm as ltb
model = ltb.LGBMClassifier(metric='auc', boosting_type='gbdt', objective="binary", num_leaves=31,  # max_depth=8
                           reg_alpha=1, reg_lambda=1, min_data_in_bin=5,
                           n_estimators=100)  # boosting_type = 'goss' ou 'rf'
model.fit(train_embeddings, y.loc[train_gen.ids])
y_hat_sup = model.predict_proba(test_embeddings)[:, 1]
fpr, tpr, threshold = roc_curve(y.loc[test_gen.ids], y_hat_sup)

plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1],'r--')
print(auc(fpr, tpr))

# compare with pd nodes
xy = pd.merge(nodes_company, labels_fraud['y'].clip(upper=1), how="left", left_index=True, right_index=True).fillna(0)
x = xy.drop(columns=['y'])
y = xy['y']
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(
    x,y, train_size=0.5, test_size=None, #stratify=claims_nodes
)
model = ltb.LGBMClassifier(metric='auc', boosting_type='gbdt', objective="binary", num_leaves=31,  # max_depth=8
                           reg_alpha=1, reg_lambda=1, min_data_in_bin=5,
                           n_estimators=100)  # boosting_type = 'goss' ou 'rf'
model.fit(xtrain, ytrain)
y_hat_sup = model.predict_proba(xtrain)[:, 1]
fpr, tpr, threshold = roc_curve(ytest, y_hat_sup)
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1],'r--')

# visualize

all_embeddings = emb_model.predict(hinsage_generator.flow(HG.nodes(node_type="company")))

y = y.astype("category")
trans = TSNE(n_components=2)
emb_transformed = pd.DataFrame(trans.fit_transform(all_embeddings), index=HG.nodes(node_type="company"))
emb_transformed["label"] = y
alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].cat.codes,
    cmap="jet",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title("TSNE visualization of GCN embeddings for cora dataset")
plt.show()
