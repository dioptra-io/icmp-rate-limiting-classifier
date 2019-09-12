from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import adanet
import tensorflow as tf
import numpy as np
import os
import glob
from sklearn import metrics
from tensorflow.python.data import Dataset
from Classification.plot_metrics import plot_metrics

# The random seed to use.
RANDOM_SEED = 42


# (x_train, y_train), (x_test, y_test) = (
#         tf.keras.datasets.boston_housing.load_data())

FEATURES_KEY = "x"


def create_training_input_fn(
    features, labels, batch_size, num_epochs=None, shuffle=True
):
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
        # idx = np.random.permutation(features.index)
        # raw_features = {feature_column : feature_values.reindex(idx) for feature_column, feature_values in features.iteritems()}

        raw_features = tf.convert_to_tensor(features.values, dtype=tf.float64)
        raw_targets = np.array(labels)
        ds = tf.data.Dataset.from_tensor_slices(
            ({FEATURES_KEY: raw_features}, raw_targets)
        )

        # ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
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
        raw_features = tf.convert_to_tensor(features.values, dtype=tf.float64)
        raw_targets = np.array(labels)
        ds = tf.data.Dataset.from_tensor_slices(
            ({FEATURES_KEY: raw_features}, raw_targets)
        )

        ds = ds.batch(batch_size)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


_NUM_LAYERS_KEY = "num_layers"


class _SimpleDNNBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, optimizer, layer_size, num_layers, learn_mixture_weights, seed):
        """Initializes a `_DNNBuilder`.

    Args:
      optimizer: An `Optimizer` instance for training both the subnetwork and
        the mixture weights.
      layer_size: The number of nodes to output at each hidden layer.
      num_layers: The number of hidden layers.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      seed: A random seed.

    Returns:
      An instance of `_SimpleDNNBuilder`.
    """

        self._optimizer = optimizer
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed

    def build_subnetwork(
        self,
        features,
        logits_dimension,
        training,
        iteration_step,
        summary,
        previous_ensemble=None,
    ):
        """See `adanet.subnetwork.Builder`."""

        input_layer = tf.to_float(features[FEATURES_KEY])
        kernel_initializer = tf.glorot_uniform_initializer(seed=self._seed)
        last_layer = input_layer
        for _ in range(self._num_layers):
            last_layer = tf.layers.dense(
                last_layer,
                units=self._layer_size,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer,
            )
        logits = tf.layers.dense(
            last_layer, units=logits_dimension, kernel_initializer=kernel_initializer
        )

        persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=self._measure_complexity(),
            persisted_tensors=persisted_tensors,
        )

    def _measure_complexity(self):
        """Approximates Rademacher complexity as the square-root of the depth."""
        return tf.sqrt(tf.to_float(self._num_layers))

    def build_subnetwork_train_op(
        self,
        subnetwork,
        loss,
        var_list,
        labels,
        iteration_step,
        summary,
        previous_ensemble,
    ):
        """See `adanet.subnetwork.Builder`."""
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(
        self, loss, var_list, logits, labels, iteration_step, summary
    ):
        """See `adanet.subnetwork.Builder`."""

        if not self._learn_mixture_weights:
            return tf.no_op()
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""

        if self._num_layers == 0:
            # A DNN with no hidden layers is a linear model.
            return "linear"
        return "{}_layer_dnn".format(self._num_layers)


class SimpleDNNGenerator(adanet.subnetwork.Generator):
    """Generates a two DNN subnetworks at each iteration.

  The first DNN has an identical shape to the most recently added subnetwork
  in `previous_ensemble`. The second has the same shape plus one more dense
  layer on top. This is similar to the adaptive network presented in Figure 2 of
  [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
  connections to hidden layers of networks from previous iterations.
  """

    def __init__(
        self, optimizer, layer_size=32, learn_mixture_weights=False, seed=None
    ):
        """Initializes a DNN `Generator`.

    Args:
      optimizer: An `Optimizer` instance for training both the subnetwork and
        the mixture weights.
      layer_size: Number of nodes in each hidden layer of the subnetwork
        candidates. Note that this parameter is ignored in a DNN with no hidden
        layers.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      seed: A random seed.

    Returns:
      An instance of `Generator`.
    """

        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            _SimpleDNNBuilder,
            optimizer=optimizer,
            layer_size=layer_size,
            learn_mixture_weights=learn_mixture_weights,
        )

    def generate_candidates(
        self,
        previous_ensemble,
        iteration_number,
        previous_ensemble_reports,
        all_reports,
    ):
        """See `adanet.subnetwork.Generator`."""

        num_layers = 0
        seed = self._seed
        if previous_ensemble:
            num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1].subnetwork.persisted_tensors[
                    _NUM_LAYERS_KEY
                ]
            )
        if seed is not None:
            seed += iteration_number
        return [
            self._dnn_builder_fn(num_layers=num_layers, seed=seed),
            self._dnn_builder_fn(num_layers=num_layers + 1, seed=seed),
        ]


# @title AdaNet parameters
# LEARNING_RATE = 0.001  #@param {type:"number"}
TRAIN_STEPS = 100000  # @param {type:"integer"}
# BATCH_SIZE = 32  #@param {type:"integer"}

LEARN_MIXTURE_WEIGHTS = False  # @param {type:"boolean"}
ADANET_LAMBDA = 0  # @param {type:"number"}
BOOSTING_ITERATIONS = 5  # @param {type:"integer"}


def train(
    periods,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    labels_n,
    learning_rate,
    steps,
    batch_size,
    learn_mixture_weights=LEARN_MIXTURE_WEIGHTS,
    adanet_lambda=ADANET_LAMBDA,
):
    """Trains an `adanet.Estimator` to predict housing prices."""

    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size
    )
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size
    )
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size
    )

    classifier = adanet.Estimator(
        head=tf.contrib.estimator.multi_class_head(
            labels_n, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
        ),
        # Define the generator, which defines our search space of subnetworks
        # to train as candidates to add to the final AdaNet model.
        subnetwork_generator=SimpleDNNGenerator(
            optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate),
            learn_mixture_weights=learn_mixture_weights,
            seed=RANDOM_SEED,
        ),
        # Lambda is a the strength of complexity regularization. A larger
        # value will penalize more complex subnetworks.
        adanet_lambda=adanet_lambda,
        # # The number of train steps per iteration.
        max_iteration_steps=TRAIN_STEPS // BOOSTING_ITERATIONS,
        # max_iteration_steps= steps // periods,
        # The evaluator will evaluate the model on the full training set to
        # compute the overall AdaNet loss (train loss + complexity
        # regularization) to select the best candidate to include in the
        # final AdaNet model.
        evaluator=adanet.Evaluator(input_fn=predict_training_input_fn),
        # The report materializer will evaluate the subnetworks' metrics
        # using the full training set to generate the reports that the generator
        # can use in the next iteration to modify its search space.
        report_materializer=adanet.ReportMaterializer(
            input_fn=predict_validation_input_fn
        ),
        # Configuration for Estimators.
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=50000,
            save_summary_steps=50000,
            tf_random_seed=RANDOM_SEED,
        ),
    )

    # Train and evaluate using using the tf.estimator tooling.
    train_spec = tf.estimator.TrainSpec(
        input_fn=training_input_fn, max_steps=TRAIN_STEPS
    )
    eval_spec = tf.estimator.EvalSpec(input_fn=predict_validation_input_fn, steps=None)

    # steps_per_period = steps / periods
    #
    # print("Training model...")
    # print("LogLoss error (on validation data):")
    # training_errors = []
    # validation_errors = []
    #
    # for period in range(0, periods):
    #     # Train the model, starting from the prior state.
    #     classifier.train(
    #         input_fn=training_input_fn,
    #         steps=steps_per_period
    #     )
    #
    #
    #     # Take a break and compute probabilities.
    #     training_predictions = list(classifier.predict(input_fn= training_input_fn))
    #     training_probabilities = np.array([item['probabilities'] for item in training_predictions])
    #     training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
    #     # training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)
    #
    #     validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
    #     validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
    #     validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
    #     # validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)
    #
    #     # Compute training and validation errors.
    #     training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    #     validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    #     # Occasionally print the current loss.
    #     print("  period %02d : %0.4f" % (period, validation_log_loss))
    #     # Add the loss metrics from this period to our list.
    #     training_errors.append(training_log_loss)
    #     validation_errors.append(validation_log_loss)
    # print("Model training finished.")
    # # Remove event files to save disk space.
    # _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))
    #
    # # Calculate final predictions (not probabilities, as above).
    # final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    # final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    #
    # accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    # print("Final accuracy (on validation data): %0.3f" % accuracy)
    #
    # plot_metrics(training_errors, validation_errors, validation_targets, final_predictions)

    # return classifier
    return tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


def ensemble_architecture(result):
    """Extracts the ensemble architecture from evaluation results."""

    architecture = result["architecture/adanet/ensembles"]
    # The architecture is a serialized Summary proto for TensorBoard.
    summary_proto = tf.summary.Summary.FromString(architecture)
    return summary_proto.value[0].tensor.string_val[0]
