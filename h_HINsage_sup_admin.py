# this is experimental
from sklearn import preprocessing, feature_extraction, model_selection
import stellargraph as sg
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras import Model, optimizers, losses, metrics

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection

# from IPython.display import display, HTML
import matplotlib

rebal_subjects = pd.concat(
    [admin_subjects[admin_subjects == "Fraud"],
    admin_subjects[admin_subjects == "Legit"].sample(admin_subjects[admin_subjects == "Fraud"].shape[0])])

train_subjects_a, val_subjects_a = model_selection.train_test_split(
    rebal_subjects, train_size=0.5, test_size=None, stratify=rebal_subjects
)

target_encoding = preprocessing.LabelBinarizer()

train_targets_a = target_encoding.fit_transform(train_subjects_a) #np.array(train_subjects)
val_targets_a = target_encoding.transform(val_subjects_a)

generator = sg.mapper.HinSAGENodeGenerator(
    HG, batch_size=1000, num_samples=[10, 5], head_node_type="person"
)
train_gen = generator.flow(train_subjects_a.index, train_targets_a)
val_gen = generator.flow(val_subjects_a.index, val_targets_a)

base_model = sg.layer.HinSAGE([16, 16], generator=generator, bias=True, dropout=0.5) # Aggregator defaults to MeanHinAggregator
x_in, x_out = base_model.in_out_tensors()

prediction = layers.Dense(1, activation="sigmoid")(x_out)
model = Model(inputs=x_in, outputs=prediction)
model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.005), metrics=['AUC'])
cbs = [EarlyStopping(monitor="val_loss", mode="min", patience=4)]
history = model.fit(train_gen, validation_data=val_gen, epochs=50, verbose=1, shuffle=False, callbacks=cbs)


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

plot_loss(history, "Mean.Agg. ", 0, "auc")
plt.legend()
#matplotlib.use('TkAgg') # depends on pycharm mode
plt.show()

import tikzplotlib
tikzplotlib.save("output_files/hinsage_admin_sup.tex", encoding='utf-8')
