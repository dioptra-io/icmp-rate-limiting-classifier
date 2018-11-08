import glob
import math
import os

import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from plot_metrics import plot_metrics


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def parse_labels_and_features(dataset, label_column, features_columns):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[label_column]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset[features_columns]

    return labels, features


def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    """A custom input_fn for sending data to the estimator for training.

    Args:
      features: The training features.
      labels: The training labels.
      batch_size: Batch size to use during training.

    Returns:
      A function that returns batches of training features and labels during
      training.
    """

    def _input_fn(num_epochs=None, shuffle=True):
        # Input pipelines are reset with each call to .train(). To ensure model
        # gets a good sampling of data, even when number of steps is small, we
        # shuffle all the data before creating the Dataset object
        idx = np.random.permutation(features.index)
        raw_features = {feature_column : feature_values.reindex(idx) for feature_column, feature_values in features.iteritems()}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    """A custom input_fn for sending mnist data to the estimator for predictions.

    Args:
      features: The features to base predictions on.
      labels: The labels of the prediction examples.

    Returns:
      A function that returns features and labels for predictions.
    """

    def _input_fn():
        raw_features = {feature_column : feature_values for feature_column, feature_values in features.iteritems()}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def train_nn_classification_model(
        periods,
        classes_n,
        feature_columns,
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
      feature_columns: A list of features
      learning_rate: An `int`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `DNNClassifier` object.
    """

    # Caution: input pipelines are reset with each call to train.
    # If the number of steps is small, your model may never see most of the data.
    # So with multiple `.train` calls like this you may want to control the length
    # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
    # or since it's in-memory data, shuffle all the data in the `input_fn`.
    steps_per_period = steps / periods

    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(feature_column) for feature_column in feature_columns]

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=classes_n,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        # training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        # validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.4f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.3f" % accuracy)

    # # Output a graph of loss metrics over periods.
    # plt.ylabel("LogLoss")
    # plt.xlabel("Periods")
    # plt.title("LogLoss vs. Periods")
    # plt.plot(training_errors, label="training")
    # plt.plot(validation_errors, label="validation")
    # plt.legend()
    # plt.show()
    #
    # # Output a plot of the confusion matrix.
    # cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # # Normalize the confusion matrix by row (i.e by the number of samples
    # # in each class).
    # cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # ax = sns.heatmap(cm_normalized, cmap="bone_r")
    # ax.set_aspect(1)
    # plt.title("Confusion matrix")
    # plt.ylabel("True label")
    # plt.xlabel("Predicted label")
    # plt.show()

    plot_metrics(training_errors, validation_errors, validation_targets, final_predictions)


    return classifier