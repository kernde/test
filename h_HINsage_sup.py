# this is experimental
from sklearn import preprocessing, feature_extraction, model_selection
import stellargraph as sg
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras import Model, optimizers, losses, metrics
from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph.layer.graphsage import MeanPoolingAggregator, AttentionalAggregator, MeanAggregator
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

rebal_subjects_a = pd.concat(
    [admin_subjects[admin_subjects == "Fraud"],
    admin_subjects[admin_subjects == "Legit"].sample(admin_subjects[admin_subjects == "Fraud"].shape[0])])

train_subjects_a, val_subjects_a = model_selection.train_test_split(
    admin_subjects, train_size=0.5, test_size=None, stratify=admin_subjects
)

target_encoding = preprocessing.LabelBinarizer()

train_targets_a = target_encoding.fit_transform(train_subjects_a) #np.array(train_subjects)
val_targets_a = target_encoding.transform(val_subjects_a)

hinsage_generator = HinSAGENodeGenerator(
    HG, batch_size=1000, num_samples=[10,5], head_node_type="person"
)

hinsage_model = HinSAGE(
    layer_sizes=[32, 32], activations=["elu","sigmoid"], generator=hinsage_generator,
    bias=True, dropout=0.5 # , aggregator=MeanAggregator # check later about aggregators
)
#hinsage_acc = run_deep_graph_infomax(hinsage_model, hinsage_generator, epochs=epochs)
corrupted_generator = CorruptedGenerator(hinsage_generator)
gen = corrupted_generator.flow(HG.nodes(node_type="person"))
# hinsage_generator.flow(train_subjects.index, train_targets)
# generator.flow(train_subjects.index, train_targets, shuffle=True)
infomax = DeepGraphInfomax(hinsage_model, corrupted_generator)

x_in, x_out = infomax.in_out_tensors()

prediction = layers.Dense(1, activation="sigmoid")(x_out)
model = Model(inputs=x_in, outputs=prediction)

es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.binary_crossentropy,
    metrics=['AUC'],
)
train_gen = corrupted_generator.flow(train_subjects_a.index, train_targets_a, shuffle=True)
test_gen = corrupted_generator.flow(val_subjects_a.index, val_targets_a)
cbs = [EarlyStopping(monitor="val_loss", mode="min", patience=3)]
history = model.fit(
    train_gen, epochs=10, validation_data=test_gen, verbose=1, shuffle=False, callbacks=cbs
)
#hinsage_acc = run_deep_graph_infomax(model, train_gen, epochs=10)


#### test (seems working)
generator = sg.mapper.HinSAGENodeGenerator(
    HG, batch_size=1000, num_samples=[5, 5], head_node_type="person"
)
aze = generator.flow(admin_subjects.index, train_targets_a)
#other_generator = sg.mapper.HinSAGENodeGenerator(
#    HG, batch_size=1000, num_samples=[2, 2], head_node_type="person"
#)

corrupted_generator = sg.mapper.CorruptedGenerator(generator)
#train_corr_gen = corrupted_generator.flow(HG.nodes("person")) # to change
#test_corr_gen = corrupted_generator.flow(HG.nodes("person"))

# find a way to change 2 above to
#
target_encoding = preprocessing.LabelBinarizer()
train_targets_a = target_encoding.fit_transform(admin_subjects) #np.array(train_subjects)
val_targets_a = target_encoding.transform(admin_subjects)
#
#train_gen = corrupted_generator.flow(train_subjects.index, train_targets_a) # , shuffle=True ?
#test_gen = corrupted_generator.flow(admin_subjects.index, val_targets_a)
## or maybe transductive
# train_corr_gen = corrupted_generator.flow(HG.nodes("person"), train_gen) # to change
# test_corr_gen = corrupted_generator.flow(HG.nodes("person"), test_gen)
#

base_model = sg.layer.HinSAGE([16, 16], generator=generator)
dgi_model = sg.layer.DeepGraphInfomax(base_model, corrupted_generator)
import tensorflow as tf

#x_in, x_out = dgi_model.in_out_tensors()
x_in, x_out = base_model.in_out_tensors()
# test
prediction = layers.Dense(1, activation="sigmoid")(x_out)
model = Model(inputs=x_in, outputs=prediction)
model.compile(loss=losses.binary_crossentropy, optimizer="Adam", metrics=['AUC'])
# test end: works

#model = tf.keras.Model(inputs=x_in, outputs=x_out)
#model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")

#small_gen = corrupted_generator.flow(HG.nodes("person"))
history = model.fit(small_gen, epochs=1)


################## real working here onwards ####################
generator = sg.mapper.HinSAGENodeGenerator(
    HG, batch_size=1000, num_samples=[5, 5], head_node_type="person"
)
target_encoding = preprocessing.LabelBinarizer()
train_targets_a = target_encoding.fit_transform(admin_subjects) #np.array(train_subjects)
#val_targets_a = target_encoding.transform(admin_subjects)

aze = generator.flow(admin_subjects.index, train_targets_a)


base_model = sg.layer.HinSAGE([16, 16], generator=generator)
x_in, x_out = base_model.in_out_tensors()

prediction = layers.Dense(1, activation="sigmoid")(x_out)
model = Model(inputs=x_in, outputs=prediction)
model.compile(loss=losses.binary_crossentropy, optimizer="Adam", metrics=['AUC'])

history = model.fit(aze, epochs=1)
