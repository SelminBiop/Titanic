from __future__ import print_function

import numpy as np
import pandas as pd
from DataWrangler import DataWrangler
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

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
titanic_passenger_test_dataframe = titanic_data_wrangler.wrangle(titanic_passenger_test_dataframe)

def train_model(
        learn_rate,
        batch_size_val,
        epoch_val,
        training_examples,
        training_targets,
        val_examples,
        val_targets
):

    model = keras.Sequential([
        keras.layers.Dense(9, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(3, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l1(0.001))
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


def transform_categorical(dataframe):
    label_encoder = LabelEncoder()
    result_list = [
        np.array([keras.utils.to_categorical(dataframe["Pclass"])]),
        np.array([keras.utils.to_categorical(dataframe["Age"])]),
        np.array([keras.utils.to_categorical(dataframe["IsAlone"])]),
        np.array([keras.utils.to_categorical(dataframe["Fare"])]),
        np.array([keras.utils.to_categorical(dataframe["FamilySize"])]),
        np.array([keras.utils.to_categorical(label_encoder.fit_transform(dataframe["Sex"]))]),
        np.array([keras.utils.to_categorical(label_encoder.fit_transform(dataframe["Embarked"]))]),
        np.array([keras.utils.to_categorical(label_encoder.fit_transform(dataframe["Cabin"]))]),
        np.array([keras.utils.to_categorical(label_encoder.fit_transform(dataframe["Title"]))]),
    ]

    return result_list


titanic_train = transform_categorical(titanic_passenger_dataframe.drop("Survived", axis=1).head(head_value))
titanic_train_target = np.array([keras.utils.to_categorical(titanic_passenger_dataframe["Survived"].head(head_value))])

titanic_validation = transform_categorical(titanic_passenger_dataframe.drop("Survived", axis=1).tail(data_length - head_value))
titanic_validation_target = np.array([keras.utils.to_categorical(titanic_passenger_dataframe["Survived"].tail(data_length - head_value))])

# Play with these (you can also try to change the architecture of the DNN) and try to consistently get a good accuracy,
# auc, precision and recall score (mostly accuracy in this case)
martin = train_model(
    learn_rate=0.00005,
    batch_size_val=5,
    epoch_val=100,
    training_examples=titanic_train,
    training_targets=titanic_train_target,
    val_examples=titanic_validation,
    val_targets=titanic_validation_target
 )

pred = martin.predict(transform_categorical(titanic_passenger_test_dataframe))
print(pred)
