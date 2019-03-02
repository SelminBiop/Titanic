from __future__ import print_function

import numpy as np
import pandas as pd
from DataWrangler import DataWrangler
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from scipy import stats

PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

titanic_passenger_dataframe = pd.read_csv(PATH, sep=",")
titanic_passenger_test_dataframe = pd.read_csv(TEST_PATH, sep=",")

# Randomize the dataset examples in case of a pattern present in the data that could "break" our training and
# validation set
titanic_passenger_dataframe = titanic_passenger_dataframe.reindex(
    np.random.permutation(titanic_passenger_dataframe.index)
)

# Get a sense of what our data looks like
print(titanic_passenger_dataframe.describe())

# ToDo : All the data analyzing stuff

titanic_data_wrangler = DataWrangler()
titanic_passenger_dataframe = titanic_data_wrangler.wrangle(titanic_passenger_dataframe)

label_encoder = LabelEncoder()
# This step is broken as !"/$%
# Todo : FIX
titanic_passenger_dataframe["Age"] = keras.utils.to_categorical(titanic_passenger_dataframe["Age"])
titanic_passenger_dataframe["Pclass"] = keras.utils.to_categorical(titanic_passenger_dataframe["Pclass"])
titanic_passenger_dataframe["IsAlone"] = keras.utils.to_categorical(titanic_passenger_dataframe["IsAlone"])
titanic_passenger_dataframe["FamilySize"] = keras.utils.to_categorical(titanic_passenger_dataframe["FamilySize"])
titanic_passenger_dataframe["Fare"] = keras.utils.to_categorical(titanic_passenger_dataframe["Fare"])
titanic_passenger_dataframe["Sex"] = keras.utils.to_categorical(label_encoder.fit_transform(titanic_passenger_dataframe["Sex"]))
titanic_passenger_dataframe["Embarked"] = keras.utils.to_categorical(label_encoder.fit_transform(titanic_passenger_dataframe["Embarked"]))
titanic_passenger_dataframe["Cabin"] = keras.utils.to_categorical(label_encoder.fit_transform(titanic_passenger_dataframe["Cabin"]))
titanic_passenger_dataframe["Title"] = keras.utils.to_categorical(label_encoder.fit_transform(titanic_passenger_dataframe["Title"]))

def train_model(
        learn_rate,
        batch_size_val,
        epoch_val,
        training_examples,
        training_targets,
        val_examples,
        val_targets
):
    print(stats.describe(training_examples))
    print(stats.describe(training_targets))
    print(stats.describe(val_examples))
    print(stats.describe(val_targets))

    model = keras.Sequential([
        keras.layers.Dense(5, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l1(0.01)),
        keras.layers.Dense(3, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(0.01))
    ])

    sgd = keras.optimizers.SGD(lr=learn_rate)

    model.compile(
        optimizer=sgd,
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    model.fit(
        x=training_examples,
        y=training_targets,
        epochs=epoch_val,
        batch_size=batch_size_val
    )

    print("Model training finished")

    test_loss, test_acc = model.evaluate(x=val_examples, y=val_targets)
    print("Loss: ", test_loss)
    print("Accuracy: ", test_acc)
    return model


# We separate the dataset in a training and validation set to verify if our model generalizes well (no overfiting)
data_length = len(titanic_passenger_dataframe.index)
train_ratio = 0.8
head_value = int(data_length * train_ratio)

train_examples = titanic_passenger_dataframe.head(head_value).drop("Survived", axis=1)
print(train_examples["Age"])

# Play with these (you can also try to change the architecture of the DNN) and try to consistently get a good accuracy,
# auc, precision and recall score (mostly accuracy in this case)
train_model(
    learn_rate=0.5,
    batch_size_val=5,
    epoch_val=500,
    training_examples=train_examples.values,
    training_targets=titanic_passenger_dataframe["Survived"].head(head_value).values,
    val_examples=titanic_passenger_dataframe.tail(data_length - head_value).drop("Survived", axis=1).values,
    val_targets=titanic_passenger_dataframe["Survived"].tail(data_length - head_value).values
)
