import glob
import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from Classification.plot_metrics import plot_metrics
from matplotlib import pyplot as plt
from Classification.metrics import compute_metrics
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def print_bad_labels(data_pred_labels, data_targets, data_examples, labeled_df):
    for i in range(0, len(data_pred_labels)):
        if not data_pred_labels[i] == data_targets.iloc[i][0]:
            true_label = data_examples.iloc[i].sort_index()
            bad_label = data_pred_labels[i]
            # print validation_pred_labels
            # print str(true_label["loss_rate_dpr_c0"])
            print (true_label.filter(regex="loss_rate_dpr_c([0-9]+)", axis=0))
            print (true_label.filter(regex="loss_rate_dpr_w([0-9]+)", axis=0))
            print (true_label.filter(regex="label_c([0-9]+)", axis=0))

            print (bad_label)

            print (labeled_df.loc[data_examples.index[i]]["measurement_id"])


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
        raw_targets = np.array(labels.reindex(idx))

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
        n_classes,
        is_multilabel,
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


    if is_multilabel:
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        multilabel_head = tf.contrib.estimator.multi_label_head(n_classes=n_classes)
        # loss_reduction = tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)
        classifier = tf.contrib.estimator.DNNEstimator(
            head = multilabel_head,
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            optimizer=my_optimizer,
            config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
        )
    else:
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            n_classes=n_classes,
            optimizer=my_optimizer,
            config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
        )
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_log_loss_errors = []
    validation_log_loss_errors = []

    training_hamming_loss_errors = []
    validation_hamming_loss_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        if is_multilabel:
            training_pred_labels = np.round([item['probabilities'] for item in training_predictions])
        else:
            training_pred_labels = np.array([item['class_ids'][0] for item in training_predictions])

        # training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        if is_multilabel:
            validation_pred_labels = np.round([item['probabilities'] for item in validation_predictions])
        else:
            validation_pred_labels = np.array([item['class_ids'][0] for item in validation_predictions])
        # validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        # validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)


        # training_hamming_loss = metrics.hamming_loss(training_targets, training_pred_labels)
        # validation_hamming_loss = metrics.hamming_loss(validation_targets, validation_pred_labels)


        # print("Training hamming loss period %02d : %0.4f" % (period, training_log_loss))
        # print("Validation hamming loss period %02d : %0.4f" % (period, validation_log_loss))



        # Occasionally print the current loss.
        print("Training log loss period %02d : %0.4f" % (period, training_log_loss))
        print("Validation log loss period %02d : %0.4f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_loss_errors.append(training_log_loss)
        validation_log_loss_errors.append(validation_log_loss)


        # DEBUG Print the labels for which the prediction are incorrect
        # print "Bad labels on training set: "
        # print_bad_labels(training_pred_labels, training_targets, training_examples)

        precision, recall, accuracy, f_score = compute_metrics(training_pred_labels, training_targets)
        print ("Precision: " + str(precision))
        print ("Recall: " + str(recall))
        print ("Accuracy: " + str(accuracy))
        print("F score: " + str(f_score))



        # print "Bad labels on validation set: "
        # print_bad_labels(validation_pred_labels, validation_targets, validation_examples)

        precision, recall, accuracy, f_score = compute_metrics(validation_pred_labels, validation_targets)
        print ("Precision: " + str(precision))
        print ("Recall: " + str(recall))
        print ("Accuracy: " + str(accuracy))
        print("F score: " + str(f_score))
        # training_hamming_loss_errors.append(training_hamming_loss)
        # validation_hamming_loss_errors.append(validation_hamming_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    if is_multilabel:
        final_predictions = np.round([item['probabilities'] for item in final_predictions])
    else:
        final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    precision, recall, accuracy, f_score = compute_metrics(final_predictions, validation_targets)
    print ("Final precision: " + str(precision))
    print ("Final recall: " + str(recall))
    print ("Final accuracy: " + str(accuracy))
    print("F score: " + str(f_score))
    # roc_auc = metrics.roc_auc_score(validation_targets, final_predictions)
    # print("ROC AUC on test data: %0.3f" % roc_auc)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_log_loss_errors, label="training")
    plt.plot(validation_log_loss_errors, label="validation")
    plt.legend()
    plt.show()

    # plt.ylabel("HammingLoss")
    # plt.xlabel("Periods")
    # plt.title("LogLoss vs. Periods")
    # plt.plot(training_hamming_loss_errors, label="training")
    # plt.plot(validation_hamming_loss_errors, label="validation")
    # plt.legend()
    # plt.show()

    # Output a plot of the confusion matrix.
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

    # plot_metrics(training_errors, validation_errors, validation_targets, final_predictions)


    return classifier