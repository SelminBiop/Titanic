from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from DataWrangler import DataWrangler
from tensorflow.python.data import Dataset

PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
RESULT_PATH = "data/submission.csv"

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


def preprocess_features(dataframe):
    selected_features = dataframe[
        ["Age",
         "Pclass",
         "Sex",
         "IsAlone",
         "FamilySize",
         "Embarked",
         "Cabin",
         "Title",
         "Fare"]]
    processed_features = selected_features.copy()
    return processed_features


titanic_data_wrangler = DataWrangler()
titanic_passenger_dataframe = titanic_data_wrangler.wrangle(titanic_passenger_dataframe)

# We separate the dataset in a training and validation set to verify if our model generalizes well (no overfiting)
train_examples = preprocess_features(titanic_passenger_dataframe.head(700))
train_targets = titanic_passenger_dataframe["Survived"].head(700)

validation_examples = preprocess_features(titanic_passenger_dataframe.tail(191))
validation_targets = titanic_passenger_dataframe["Survived"].tail(191)


def my_input_fn(features, targets=None, batch_size_val=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    if targets is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, targets)

    ds = Dataset.from_tensor_slices(inputs)
    ds = ds.batch(batch_size_val).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    return ds.make_one_shot_iterator().get_next()


# We need to tranform the different features so it can be consumed by our tensorflow model. All our features are
# categorical in our case. the indicator_column was necessary for our deep neural network but not necessary with
# shallow learning algorithms
categorical_features = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Age", hash_bucket_size=7, dtype=tf.int64))] \
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Pclass", hash_bucket_size=3, dtype=tf.int64))] \
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Sex", hash_bucket_size=2))] \
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("IsAlone", hash_bucket_size=2, dtype=tf.int64))]\
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("FamilySize", hash_bucket_size=11, dtype=tf.int64))]\
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Embarked", hash_bucket_size=3))]\
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Cabin", hash_bucket_size=9))]\
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Title", hash_bucket_size=6))]\
                       + [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket("Fare", hash_bucket_size=4, dtype=tf.int64))]


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
    hidden_units = [9, 3, 1]

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


def evaluate_model(model, print_results=False):
    evaluation_metrics = model.evaluate(input_fn=lambda: my_input_fn(
        validation_examples,
        validation_targets,
        num_epochs=1,
        shuffle=False
    ))

    if print_results:
        print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
        print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
        print(evaluation_metrics)

    return evaluation_metrics


# Play with these (you can also try to change the architecture of the DNN) and try to consistently get a good accuracy,
# auc, precision and recall score (mostly accuracy in this case)

linear_classifier = train_model(
    learn_rate=0.00005,
    steps_val=500000,
    batch_size_val=5,
    training_examples=train_examples,
    training_targets=train_targets,
    val_examples=validation_examples,
    val_targets=validation_targets
)

evaluate_model(
    model=linear_classifier,
    print_results=True
)

test_passenger_ids = titanic_passenger_test_dataframe["PassengerId"]
titanic_passenger_test_dataframe = titanic_data_wrangler.wrangle(titanic_passenger_test_dataframe)
results = linear_classifier.predict(
    input_fn=lambda: my_input_fn(
        features=preprocess_features(titanic_passenger_test_dataframe),
        num_epochs=1,
        shuffle=False
    )
)
results = [result["class_ids"][0] for result in results]
submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": results
})
submission.to_csv(RESULT_PATH, index=False)
