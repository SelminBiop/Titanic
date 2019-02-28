from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

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

# Some age entries are missing so we'll replace them with the average
# ToDo : Instead of using the mean, we could try to get an average age from Titles present in the names and assign
#  these age values instead (separate people in categories)
mean_age = np.mean(titanic_passenger_dataframe["Age"])
titanic_passenger_dataframe["Age"].fillna(mean_age, inplace=True)

# Also, let's transform them into age categories. It makes more sense that way
# ToDo : After careful data analysis try to find best age categories
titanic_passenger_dataframe["Age"] = titanic_passenger_dataframe["Age"].astype(np.int64)
titanic_passenger_dataframe["Age"] = titanic_passenger_dataframe["Age"].apply(
    lambda age:
    1 if age < 10 else
    2 if 10 <= age < 18 else
    3 if 18 <= age < 25 else
    4 if 25 <= age < 30 else
    5 if 30 <= age < 35 else
    6 if 35 <= age < 45 else
    7
)

# Some Embarked entries are missing, let's try and replace them with where most of the people embarked from
# ToDo : Would it be possible to do something similar to Age?
titanic_passenger_dataframe["Embarked"].fillna(
    titanic_passenger_dataframe["Embarked"].value_counts().index[0],
    inplace=True
)

# The cabin (first letter) can be indicative of where someone was situated, close to a life boat maybe?
# We fill with U for unknown
titanic_passenger_dataframe["Cabin"].fillna("U", inplace=True)
titanic_passenger_dataframe["Cabin"] = titanic_passenger_dataframe["Cabin"].apply(lambda cabin: cabin[0])

# Let's engineer two new features, FamilySize and if the passenger was alone or not (could influence how easy it was
# to find a place on a boat for everyone)
titanic_passenger_dataframe["FamilySize"] = titanic_passenger_dataframe["SibSp"] + titanic_passenger_dataframe["Parch"]

# IsAlone is a bit more complicated since a child could be accompanied by a nanny or an aunt but still have a
# FamilySize of 0, we'll assume that everyone under the age of 18 was accompanied
titanic_passenger_dataframe["IsAlone"] = titanic_passenger_dataframe["FamilySize"].apply(lambda fam_size: 0 if fam_size > 0 else 1)
is_old_enough = titanic_passenger_dataframe["Age"].apply(lambda age: 0 if age < 18 else 1)
titanic_passenger_dataframe["IsAlone"] = titanic_passenger_dataframe["IsAlone"] * is_old_enough

# Transformation for the benefit of Tensorflow
titanic_passenger_dataframe["Pclass"].apply(np.int64)

# ToDo : All the transformations should be applied on the test set (define a function)
titanic_passenger_test_dataframe["Age"].fillna(mean_age, inplace=True)
titanic_passenger_test_dataframe["Pclass"].apply(np.int64)
titanic_passenger_test_dataframe["Age"] = titanic_passenger_dataframe["Age"].astype(np.int64)


def preprocess_features(dataframe):
    selected_features = dataframe[
        ["Age",
         "Pclass",
         "Sex",
         "IsAlone",
         "FamilySize"]]
    processed_features = selected_features.copy()
    return processed_features


# We separate the dataset in a training and validation set to verify if our model generalizes well (no overfiting)
train_examples = preprocess_features(titanic_passenger_dataframe.head(700))
train_targets = titanic_passenger_dataframe["Survived"].head(700)

validation_examples = preprocess_features(titanic_passenger_dataframe.tail(191))
validation_targets = titanic_passenger_dataframe["Survived"].tail(191)


def my_input_fn(features, targets, batch_size_val=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size_val).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# We need to tranform the different features so it can be consumed by our tensorflow model. All our features are
# categorical in our case. the indicator_column was necessary for our deep neural network but not necessary with
# shallow learning algorithms
categorical_features = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Age", hash_bucket_size=7, dtype=tf.int64))] \
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Pclass", hash_bucket_size=3, dtype=tf.int64))] \
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Sex", hash_bucket_size=2))] \
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("IsAlone", hash_bucket_size=2, dtype=tf.int64))]\
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("FamilySize", hash_bucket_size=20, dtype=tf.int64))]
                       #+ [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Embarked", hash_bucket_size=3))]\
                       #+ [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Cabin", hash_bucket_size=9))]


def train_model(
        learn_rate,
        steps_val,
        batch_size_val,
        training_examples,
        training_targets,
        val_examples,
        val_targets
):
    periods = 10
    steps_per_period = steps_val / periods

# Architecture of our Neural Network
    hidden_units = [2, 4, 7]

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# ToDo : Try keras to more easily add Regularization and PCA to avoid overfitting and perform dimension reduction
    linear_classifier = tf.estimator.DNNClassifier(
        n_classes=2,
        hidden_units=hidden_units,
        feature_columns=categorical_features,
        optimizer=my_optimizer
    )

    training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets,
        batch_size_val=batch_size_val
    )
    predict_training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets,
        num_epochs=1,
        shuffle=False
    )
    predict_validation_input_fn = lambda: my_input_fn(
        val_examples,
        val_targets,
        num_epochs=1,
        shuffle=False
    )

    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'][0] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'][0] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(val_targets, validation_probabilities)

        print("  period %02d : %0.2f" % (period, training_log_loss))

        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished")

    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    plt.show()

    return linear_classifier


def evaluate_model(learn_rate, steps_val, batch_size_val, print_results=False):
    linear_classifier = train_model(
        learn_rate=learn_rate,
        steps_val=steps_val,
        batch_size_val=batch_size_val,
        training_examples=train_examples,
        training_targets=train_targets,
        val_examples=validation_examples,
        val_targets=validation_targets
    )

    evaluation_metrics = linear_classifier.evaluate(input_fn=lambda: my_input_fn(
        validation_examples,
        validation_targets,
        num_epochs=1,
        shuffle=False
    ))

    if print_results:
        print("Learning rate: %0.5f, Steps: %d, Batch size: %d" % (learn_rate, steps_val, batch_size_val))
        print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
        print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
        print(evaluation_metrics)

    return evaluation_metrics


# Play with these (you can also try to change the architecture of the DNN) and try to consistently get a good accuracy,
# auc, precision and recall score (mostly accuracy in this case)
evaluate_model(
    learn_rate=0.00003,
    steps_val=500000,
    batch_size_val=5,
    print_results=True
)
