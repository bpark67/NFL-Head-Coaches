from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

dftrain = pd.read_csv("NFL Head Coaches_train.csv")
dfeval = pd.read_csv("NFL Head Coaches_eval.csv")
dftrain.pop("From")
dftrain.pop("To")
dftrain.pop("G")
dftrain.pop("W")
dftrain.pop("L")
dftrain.pop("T")
dftrain.pop("W-L")
dftrain.pop("Standard WP")
dftrain.pop("G plyf")
dftrain.pop("W plyf")
dftrain.pop("L plyf")
dftrain.pop("Chmp")
dftrain.pop("Team")
dftrain.pop("Tenure")
dftrain.pop("Playoffs")
dftrain.pop("SB")
dfeval.pop("From")
dfeval.pop("To")
dfeval.pop("G")
dfeval.pop("W")
dfeval.pop("L")
dfeval.pop("T")
dfeval.pop("W-L")
dfeval.pop("Standard WP")
dfeval.pop("G plyf")
dfeval.pop("W plyf")
dfeval.pop("L plyf")
dfeval.pop("Chmp")
dfeval.pop("Team")
dfeval.pop("Tenure")
dfeval.pop("Playoffs")
dfeval.pop("SB")
y_train = dftrain.pop("Current")
y_eval = dfeval.pop("Current")

NUMERIC_COLUMNS = ["W-L%"]

feature_columns = []
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(
        tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
    )


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df)
        )  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(
            num_epochs
        )  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


train_input_fn = make_input_fn(
    dftrain, y_train
)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(
    eval_input_fn
)  # get model metrics/stats by testing on tetsing data

print(result["accuracy"])

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])
print(y_eval.loc[0])
print(result[0]["probabilities"][1])

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred["probabilities"][1] for pred in pred_dicts])

# probs.plot(kind="hist", bins=20, title="predicted probabilities")
# plt.show()

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title("ROC curve")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.xlim(
    0,
)
plt.ylim(
    0,
)
plt.show()
