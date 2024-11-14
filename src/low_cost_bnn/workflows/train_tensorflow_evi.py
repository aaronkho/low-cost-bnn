import os
import argparse
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from ..utils.pipeline_tools import (
    setup_logging,
    print_settings,
    preprocess_data
)
from ..utils.helpers_tensorflow import (
    default_dtype,
    default_device,
    get_device_info,
    set_device_parallelism,
    set_tf_logging_level,
    create_data_loader,
    create_scheduled_adam_optimizer,
    create_regressor_model,
    create_regressor_loss_function,
    wrap_regressor_model,
    save_model
)

logger = logging.getLogger("train_tensorflow")


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Path and name of input HDF5 file containing training data set')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of output HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of output file to store training metrics')
    parser.add_argument('--input_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--input_trim', metavar='val', type=float, default=None, help='Normalized limit beyond which can be considered outliers from input variable trimming')
    parser.add_argument('--output_trim', metavar='val', type=float, default=None, help='Normalized limit beyond which can be considered outliers from output variable trimming')
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--data_split_file', metavar='path', type=str, default=None, help='Optional path and name of output HDF5 file of training, validation, and test dataset split indices')
    parser.add_argument('--max_epoch', metavar='n', type=int, default=100000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=50, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--minimum_performance', metavar='val', type=float, nargs='*', default=None, help='Set minimum value in adjusted R-squared per output before early stopping is activated')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--generalized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the generalized hidden layers')
    parser.add_argument('--specialized_layer', metavar='n', type=int, nargs='*', default=None, help='Number of specialized hidden layers, given for each output')
    parser.add_argument('--specialized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layers, sequential per output stack')
    parser.add_argument('--l1_reg_general', metavar='wgt', type=float, default=0.2, help='L1 regularization parameter used in the generalized hidden layers')
    parser.add_argument('--l2_reg_general', metavar='wgt', type=float, default=0.8, help='L2 regularization parameter used in the generalized hidden layers')
    parser.add_argument('--rel_reg_special', metavar='wgt', type=float, default=0.1, help='Relative regularization used in the specialized hidden layers compared to the generalized layers')
    parser.add_argument('--nll_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the NLL loss term')
    parser.add_argument('--evi_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the evidential loss term')
    parser.add_argument('--reg_weight', metavar='wgt', type=float, default=0.01, help='Weight to apply to regularization loss term')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.95, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epoch', metavar='n', type=float, default=10, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to output log file where script related print outs will be stored')
    parser.add_argument('--checkpoint_freq', metavar='n', type=int, default=0, help='Number of epochs between saves of model checkpoint')
    parser.add_argument('--checkpoint_dir', metavar='path', type=str, default=None, help='Optional path to directory where checkpoints will be saved')
    parser.add_argument('--save_initial', default=False, action='store_true', help='Toggle on saving of initialized model before any training, for debugging')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


@tf.function
def train_tensorflow_evidential_step(
    model,
    optimizer,
    loss_function,
    feature_batch,
    target_batch,
    reg_weight,
    dataset_size,
    training=True,
    verbosity=0
):

    n_inputs = model.n_inputs
    n_outputs = model.n_outputs

    replica_context = tf.distribute.get_replica_context()
    if replica_context is not None:
        batch_size = tf.cast(tf.reduce_sum(replica_context.all_gather(tf.stack([tf.shape(feature_batch)], axis=0), axis=0), axis=0)[0], dtype=default_dtype)
    else:
        batch_size = tf.cast(feature_shape[0], dtype=default_dtype)

    # Set up training targets into a single large tensor
    target_values = tf.stack([target_batch, tf.zeros(tf.shape(target_batch), dtype=default_dtype), tf.zeros(tf.shape(target_batch), dtype=default_dtype), tf.zeros(tf.shape(target_batch), dtype=default_dtype)], axis=1)
    batch_loss_targets = tf.stack([target_values, target_values], axis=2)

    with tf.GradientTape() as tape:

        # For mean data inputs, e.g. training data
        outputs = model(feature_batch, training=training)

        # Acquire regularization loss after evaluation of network
        model_metrics = model.get_metrics_result()
        step_regularization_loss = tf.constant(0.0, dtype=default_dtype)
        if 'regularization_loss' in model_metrics:
            step_regularization_loss = tf.math.multiply(tf.constant(reg_weight, dtype=default_dtype), model_metrics['regularization_loss'])
        # Regularization loss is invariant on batch size, but this improves comparative context in metrics
        step_regularization_loss = tf.math.divide(tf.math.multiply(step_regularization_loss, batch_size), dataset_size)

        if training and tf.executing_eagerly() and verbosity >= 4:
            for ii in range(n_outputs):
                logger.debug(f'     gamma: {outputs[0, 0, ii]}')
                logger.debug(f'     nu: {outputs[0, 1, ii]}')
                logger.debug(f'     alpha: {outputs[0, 2, ii]}')
                logger.debug(f'     beta: {outputs[0, 3, ii]}')

        # Set up network predictions into equal shape tensor as training targets
        batch_loss_predictions = tf.stack([outputs, outputs], axis=2)
        if n_outputs == 1:
            batch_loss_targets = tf.squeeze(batch_loss_targets, axis=-1)
            batch_loss_predictions = tf.squeeze(batch_loss_predictions, axis=-1)

        # Compute total loss to be used in adjusting weights and biases
        step_total_loss = loss_function(batch_loss_targets, batch_loss_predictions)
        step_total_loss = tf.math.add(step_total_loss, step_regularization_loss)
        adjusted_step_total_loss = tf.math.divide(step_total_loss, batch_size)

        # Remaining loss terms purely for inspection purposes
        step_likelihood_loss = loss_function._calculate_likelihood_loss(
            tf.squeeze(tf.gather(batch_loss_targets, indices=[0], axis=2), axis=2),
            tf.squeeze(tf.gather(batch_loss_predictions, indices=[0], axis=2), axis=2)
        )
        step_evidential_loss = loss_function._calculate_evidential_loss(
            tf.squeeze(tf.gather(batch_loss_targets, indices=[1], axis=2), axis=2),
            tf.squeeze(tf.gather(batch_loss_predictions, indices=[1], axis=2), axis=2)
        )

    # Apply back-propagation
    if training:
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(adjusted_step_total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

    return (
        tf.reshape(step_total_loss, shape=(-1, 1)),
        tf.reshape(step_regularization_loss, shape=(-1, 1)),
        tf.reshape(step_likelihood_loss, shape=(-1, n_outputs)),
        tf.reshape(step_evidential_loss, shape=(-1, n_outputs))
    )


@tf.function
def distributed_train_tensorflow_evidential_step(
    strategy,
    model,
    optimizer,
    loss_function,
    feature_batch,
    target_batch,
    reg_weight,
    dataset_size,
    training=True,
    verbosity=0
):

    replica_total_loss, replica_regularization_loss, replica_likelihood_loss, replica_evidential_loss = strategy.run(
        train_tensorflow_evidential_step,
        args=(model, optimizer, loss_function, feature_batch, target_batch, reg_weight, dataset_size, training, verbosity)
    )
    return (
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_total_loss, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_regularization_loss, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_likelihood_loss, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_evidential_loss, axis=0)
    )


@tf.function
def train_tensorflow_evidential_epoch(
    strategy,
    model,
    optimizer,
    dataloader,
    loss_function,
    reg_weight,
    training=True,
    dataset_length=None,
    verbosity=0
):

    # Using the None option here is unwieldy for large datasets, recommended to always pass in correct length
    dataset_size = tf.cast(dataloader.unbatch().cardinality(), dtype=default_dtype) if dataset_length is None else tf.constant(dataset_length, dtype=default_dtype)
    n_outputs = model.n_outputs

    step_total_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'total_loss_array')
    step_regularization_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'reg_loss_array')
    step_likelihood_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'nll_loss_array')
    step_evidential_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'evi_loss_array')

    # Training loop through minibatches - each loop pass is one step
    nn = 0
    for feature_batch, target_batch in dataloader:

        # Evaluate training step on batch using distribution strategy
        step_total_loss, step_regularization_loss, step_likelihood_loss, step_evidential_loss = distributed_train_tensorflow_evidential_step(
            strategy,
            model,
            optimizer,
            loss_function,
            feature_batch,
            target_batch,
            reg_weight,
            dataset_size,
            training=training,
            verbosity=verbosity
        )

        # Accumulate batch losses to determine epoch loss
        fill_index = tf.cast(nn + 1, tf.int32)
        step_total_losses = step_total_losses.write(fill_index, tf.reshape(step_total_loss, shape=[-1, 1]))
        step_regularization_losses = step_regularization_losses.write(fill_index, tf.reshape(step_regularization_loss, shape=[-1, 1]))
        step_likelihood_losses = step_likelihood_losses.write(fill_index, tf.reshape(step_likelihood_loss, shape=[-1, n_outputs]))
        step_evidential_losses = step_evidential_losses.write(fill_index, tf.reshape(step_evidential_loss, shape=[-1, n_outputs]))

        nn += 1

    epoch_total_loss = tf.reduce_sum(step_total_losses.concat(), axis=0)
    epoch_regularization_loss = tf.reduce_sum(step_regularization_losses.concat(), axis=0)
    epoch_likelihood_loss = tf.reduce_sum(step_likelihood_losses.concat(), axis=0)
    epoch_evidential_loss = tf.reduce_sum(step_evidential_losses.concat(), axis=0)

    return (
        epoch_total_loss,
        epoch_regularization_loss,
        epoch_likelihood_loss,
        epoch_evidential_loss
    )


def meter_tensorflow_evidential_epoch(
    model,
    inputs,
    targets,
    losses,
    loss_trackers={},
    performance_trackers={},
    dataset_length=None,
    verbosity=0
):

    n_inputs = model.n_inputs
    n_outputs = model.n_outputs
    dataset_size = inputs.shape[0] if dataset_length is None else dataset_length
    total_loss, reg_loss, nll_loss, evi_loss = losses

    outputs = model(inputs, training=False)
    means = tf.squeeze(tf.gather(outputs, indices=[0], axis=1), axis=1)

    loss_trackers['total'].update_state(total_loss / dataset_size)
    total_metric = loss_trackers['total'].result()

    loss_trackers['reg'].update_state(reg_loss / dataset_size)
    reg_metric = loss_trackers['reg'].result()

    nll_metric = [np.nan] * n_outputs
    evi_metric = [np.nan] * n_outputs
    adjr2_metric = [np.nan] * n_outputs
    mae_metric = [np.nan] * n_outputs
    mse_metric = [np.nan] * n_outputs

    for ii in range(n_outputs):

        metric_targets = np.atleast_2d(targets[:, ii]).T
        metric_results = np.atleast_2d(means[:, ii].numpy()).T

        loss_trackers['nll'][ii].update_state(nll_loss[ii] / dataset_size)
        nll_metric[ii] = loss_trackers['nll'][ii].result()

        loss_trackers['evi'][ii].update_state(evi_loss[ii] / dataset_size)
        evi_metric[ii] = loss_trackers['evi'][ii].result()

        performance_trackers['adjr2'][ii].update_state(metric_targets, metric_results)
        r2 = performance_trackers['adjr2'][ii].result()
        factor = tf.constant((float(dataset_size) - 1.0) / (float(dataset_size) - float(n_inputs) - 1.0), dtype=r2.dtype)
        ones = tf.constant(1.0, dtype=r2.dtype)
        adjr2_metric[ii] = tf.subtract(ones, tf.multiply(tf.subtract(ones, r2), factor))

        performance_trackers['mae'][ii].update_state(metric_targets, metric_results)
        mae_metrics[ii] = performance_trackers['mae'][ii].result().numpy().tolist()

        performance_trackers['mse'][ii].update_state(metric_targets, metric_results)
        mse_metrics[ii] = performance_trackers['mse'][ii].result().numpy().tolist()

    nll_metric = tf.stack(nll_metric, axis=0)
    evi_metric = tf.stack(evi_metric, axis=0)
    adjr2_metric = tf.stack(adjr2_metric, axis=0)
    mae_metric = tf.stack(mae_metric, axis=0)
    mse_metric = tf.stack(mse_metric, axis=0)

    return (
        tf.stack([total_metric], axis=0),
        tf.stack([reg_metric], axis=0),
        tf.stack([nll_metric], axis=0),
        tf.stack([evi_metric], axis=0),
        tf.stack([adjr2_metric], axis=0),
        tf.stack([mae_metric], axis=0),
        tf.stack([mse_metric], axis=0)
    )


def distributed_meter_tensorflow_evidential_epoch(
    strategy,
    model,
    inputs,
    targets,
    losses,
    loss_trackers={},
    performance_trackers={},
    dataset_length=None,
    verbosity=0
):

    replica_total_metric, replica_reg_metric, replica_nll_metric, replica_evi_metric, replica_adjr2_metric, replica_mae_metric, replica_mse_metric = strategy.run(
        meter_tensorflow_evidential_epoch,
        args=(model, inputs, targets, losses, loss_trackers, performance_trackers, dataset_length, verbosity)
    )

    return (
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_total_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_reg_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_nll_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_evi_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_adjr2_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_mae_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_mse_metric, axis=0)
    )


def train_tensorflow_evidential(
    strategy,
    model,
    optimizer,
    features_train,
    targets_train,
    features_valid,
    targets_valid,
    loss_function,
    reg_weight,
    max_epochs,
    batch_size=None,
    patience=None,
    r2_minimums=None,
    seed=None,
    checkpoint_freq=0,
    checkpoint_path=None,
    features_scaler=None,
    targets_scaler=None,
    verbosity=0
):

    n_inputs = features_train.shape[-1]
    n_outputs = targets_train.shape[-1]
    train_length = features_train.shape[0]
    valid_length = features_valid.shape[0]
    n_no_improve = 0
    improve_tol = 0.0
    r2_thresholds = None
    if isinstance(r2_minimums, (list, tuple, np.ndarray)):
        r2_thresholds = [-1.0] * n_outputs
        for ii in range(n_outputs):
            r2_thresholds[ii] = float(r2_minimums[ii]) if ii < len(r2_minimums) else -1.0

    if verbosity >= 2:
        logger.info(f' Number of inputs: {n_inputs}')
        logger.info(f' Number of outputs: {n_outputs}')
        logger.info(f' Training set size: {train_length}')
        logger.info(f' Validation set size: {valid_length}')

    # Create data loaders, including minibatching for training set
    train_data = (
        features_train.astype(default_dtype),
        targets_train.astype(default_dtype)
    )
    valid_data = (
        features_valid.astype(default_dtype),
        targets_valid.astype(default_dtype)
    )
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data, batch_size=valid_length)

    train_loader = strategy.experimental_distribute_dataset(train_loader)
    valid_loader = strategy.experimental_distribute_dataset(valid_loader)

    with strategy.scope():

        # Create training tracker objects to facilitate external analysis of pipeline
        train_loss_trackers = {
            'total': tf.keras.metrics.Sum(name=f'train_total', dtype=default_dtype),
            'reg': tf.keras.metrics.Sum(name=f'train_reg', dtype=default_dtype),
            'nll': [],
            'evi': [],
        }
        for ii in range(n_outputs):
            train_loss_trackers['nll'].append(tf.keras.metrics.Sum(name=f'train_likelihood{ii}', dtype=default_dtype))
            train_loss_trackers['evi'].append(tf.keras.metrics.Sum(name=f'train_evidential{ii}', dtype=default_dtype))

        train_performance_trackers = {
            'adjr2': [],
            'mae': [],
            'mse': [],
        }
        for ii in range(n_outputs):
            train_performance_trackers['adjr2'].append(tf.keras.metrics.R2Score(num_regressors=0, name=f'train_r2{ii}', dtype=default_dtype))
            train_performance_trackers['mae'].append(tf.keras.metrics.MeanAbsoluteError(name=f'train_mae{ii}', dtype=default_dtype))
            train_performance_trackers['mse'].append(tf.keras.metrics.MeanSquaredError(name=f'train_mse{ii}', dtype=default_dtype))

        # Create validation tracker objects to facilitate external analysis of pipeline
        valid_loss_trackers = {
            'total': tf.keras.metrics.Sum(name=f'valid_total', dtype=default_dtype),
            'reg': tf.keras.metrics.Sum(name=f'valid_reg', dtype=default_dtype),
            'nll': [],
            'evi': [],
        }
        for ii in range(n_outputs):
            valid_loss_trackers['nll'].append(tf.keras.metrics.Sum(name=f'valid_likelihood{ii}', dtype=default_dtype))
            valid_loss_trackers['evi'].append(tf.keras.metrics.Sum(name=f'valid_evidential{ii}', dtype=default_dtype))

        valid_performance_trackers = {
            'adjr2': [],
            'mae': [],
            'mse': [],
        }
        for ii in range(n_outputs):
            valid_performance_trackers['adjr2'].append(tf.keras.metrics.R2Score(num_regressors=0, name=f'valid_r2{ii}', dtype=default_dtype))
            valid_performance_trackers['mae'].append(tf.keras.metrics.MeanAbsoluteError(name=f'valid_mae{ii}', dtype=default_dtype))
            valid_performance_trackers['mse'].append(tf.keras.metrics.MeanSquaredError(name=f'valid_mse{ii}', dtype=default_dtype))

    # Output metrics containers
    total_train_list = []
    reg_train_list = []
    nll_train_list = []
    evi_train_list = []
    r2_train_list = []
    mae_train_list = []
    mse_train_list = []
    total_valid_list = []
    reg_valid_list = []
    nll_valid_list = []
    evi_valid_list = []
    r2_valid_list = []
    mae_valid_list = []
    mse_valid_list = []

    # Output container for the best trained model
    best_validation_loss = None
    best_model = tf.keras.models.clone_model(model)
    best_model.set_weights(model.get_weights())

    # Training loop
    stop_requested = False
    thresholds_surpassed = [False] * n_outputs if isinstance(r2_thresholds, list) else [True] * n_outputs
    current_thresholds_surpassed = [False] * n_outputs if isinstance(r2_thresholds, list) else [True] * n_outputs
    for epoch in range(max_epochs):

        # Training routine described in here
        train_losses = train_tensorflow_evidential_epoch(
            strategy,
            model,
            optimizer,
            train_loader,
            loss_function,
            reg_weight,
            training=True,
            dataset_length=train_length,
            verbosity=verbosity
        )

        # Evaluate model with full training data set for performance tracking
        train_metrics = distributed_meter_tensorflow_evidential_epoch(
            strategy,
            model,
            train_data[0],
            train_data[1],
            train_losses,
            loss_trackers=train_loss_trackers,
            performance_trackers=train_performance_trackers,
            dataset_length=train_length,
            verbosity=verbosity
        )
        train_total, train_reg, train_nll, train_evi, train_adjr2, train_mae, train_mse = train_metrics

        total_train_list.append(train_total.numpy().tolist())
        reg_train_list.append(train_reg.numpy().tolist())
        nll_train_list.append(train_nll.numpy().tolist())
        evi_train_list.append(train_evi.numpy().tolist())
        r2_train_list.append(train_adjr2.numpy().tolist())
        mae_train_list.append(train_mae.numpy().tolist())
        mse_train_list.append(train_mse.numpy().tolist())

        # Reuse training routine to evaluate validation data
        valid_losses = train_tensorflow_evidential_epoch(
            strategy,
            model,
            optimizer,
            valid_loader,
            loss_function,
            reg_weight,
            training=False,
            dataset_length=valid_length,
            verbosity=verbosity
        )

        # Evaluate model with full validation data set for performance tracking
        valid_metrics = distributed_meter_tensorflow_evidential_epoch(
            strategy,
            model,
            valid_data[0],
            valid_data[1],
            valid_losses,
            loss_trackers=valid_loss_trackers,
            performance_trackers=valid_performance_trackers,
            dataset_length=valid_length,
            verbosity=verbosity
        )
        valid_total, valid_reg, valid_nll, valid_evi, valid_adjr2, valid_mae, valid_mse = valid_metrics

        total_valid_list.append(valid_total.numpy().tolist())
        reg_valid_list.append(valid_reg.numpy().tolist() * float(valid_length) / float(train_length))  # Invariant to batch size, needed for comparison
        nll_valid_list.append(valid_nll.numpy().tolist())
        evi_valid_list.append(valid_evi.numpy().tolist())
        r2_valid_list.append(valid_adjr2.numpy().tolist())
        mae_valid_list.append(valid_mae.numpy().tolist())
        mse_valid_list.append(valid_mse.numpy().tolist())

        # Enable early stopping routine if minimum performance threshold is met
        if isinstance(r2_thresholds, list) and not all(current_thresholds_surpassed):
            individual_minimum_flag = True if all(thresholds_surpassed) else False
            if not np.all(np.isfinite(r2_valid_list[-1])):
                for ii in range(len(thresholds_surpassed)):
                    thresholds_surpassed[ii] = True
                    current_thresholds_surpassed[ii] = True
                logger.warning(f'An adjusted R-squared value of NaN was detected, enabling early stopping to prevent large computational waste...')
            else:
                for ii in range(n_outputs):
                    if r2_valid_list[-1][ii] >= r2_thresholds[ii]:
                        if not thresholds_surpassed[ii] and r2_thresholds[ii] >= 0.0:
                            logger.info(f'Requested minimum performance on Output {ii} of {r2_thresholds[ii]:.5f} exceeded at epoch {epoch + 1}')
                        thresholds_surpassed[ii] = True
                        current_thresholds_surpassed[ii] = True
                    else:
                        current_thresholds_surpassed[ii] = False
            if all(thresholds_surpassed) and not individual_minimum_flag:
                logger.info(f'** All requested minimum performances individually exceeded at epoch {epoch + 1} **')
            if all(current_thresholds_surpassed):
                logger.info(f'** All requested minimum performances simultaneously exceeded at epoch {epoch + 1} **')

        # Save model into output container if it is the best so far
        simultaneous_minimum_flag = True
        enable_patience = all(current_thresholds_surpassed) if simultaneous_minimum_flag else all(thresholds_surpassed)
        if enable_patience:
            if best_validation_loss is None:
                best_validation_loss = total_valid_list[-1] + improve_tol + 1.0e-3
            n_no_improve = n_no_improve + 1 if best_validation_loss < (total_valid_list[-1] + improve_tol) else 0
            if n_no_improve == 0:
                best_validation_loss = total_valid_list[-1]
                best_model.set_weights(model.get_weights())

        # Request training stop if early stopping is enabled
        if isinstance(patience, int) and patience > 0 and n_no_improve >= patience:
            stop_requested = True

        print_per_epochs = 100
        if verbosity >= 2:
            print_per_epochs = 10
        if verbosity >= 3:
            print_per_epochs = 1
        if (epoch + 1) % print_per_epochs == 0:
            epoch_str = f'Epoch {epoch + 1}:'
            logger.info(f' {epoch_str} Train -- total_train = {total_train_list[-1]:.3f}, reg_train = {reg_train_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.info(f'  -> Output {ii}: r2 = {r2_train_list[-1][ii]:.3f}, mse = {mse_train_list[-1][ii]:.3f}, mae = {mae_train_list[-1][ii]:.3f}, nll = {nll_train_list[-1][ii]:.3f}, evi = {evi_train_list[-1][ii]:.3f}')
            logger.info(f' {epoch_str} Valid -- total_valid = {total_valid_list[-1]:.3f}, reg_valid = {reg_valid_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.info(f'  -> Output {ii}: r2 = {r2_valid_list[-1][ii]:.3f}, mse = {mse_valid_list[-1][ii]:.3f}, mae = {mae_valid_list[-1][ii]:.3f}, nll = {nll_valid_list[-1][ii]:.3f}, evi = {evi_valid_list[-1][ii]:.3f}')

        # Model Checkpoint
        # ------------------------------------------------
        if checkpoint_path is not None and checkpoint_freq > 0:
            if (epoch + 1) % checkpoint_freq == 0:
                check_path = checkpoint_path / f'checkpoint_model_epoch{epoch+1}.keras'
                checkpoint_model = tf.keras.models.clone_model(model)
                checkpoint_model.set_weights(model.get_weights())
                if features_scaler is not None and targets_scaler is not None:
                    checkpoint_model = wrap_regressor_model(checkpoint_model, features_scaler, targets_scaler)
                save_model(checkpoint_model, check_path)

                checkpoint_metrics_dict = {
                    'train_total': total_train_list,
                    'valid_total': total_valid_list,
                    'train_reg': reg_train_list,
                    'train_r2': r2_train_list,
                    'train_mse': mse_train_list,
                    'train_mae': mae_train_list,
                    'train_nll': nll_train_list,
                    'train_evi': evi_train_list,
                    'valid_reg': reg_valid_list,
                    'valid_r2': r2_valid_list,
                    'valid_mse': mse_valid_list,
                    'valid_mae': mae_valid_list,
                    'valid_nll': nll_valid_list,
                    'valid_evi': evi_valid_list,
                }

                checkpoint_dict = {}
                for key, val in checkpoint_metrics_dict.items():
                    if key.endswith('total') or key.endswith('reg'):
                        metric = np.array(val)
                        checkpoint_dict[f'{key}'] = metric.flatten()
                    else:
                        metric = np.atleast_2d(val)
                        for xx in range(n_outputs):
                            checkpoint_dict[f'{key}{xx}'] = metric[:, xx].flatten()
                checkpoint_metrics_df = pd.DataFrame(data=checkpoint_dict)

                checkpoint_metrics_path = checkpoint_path / f'checkpoint_metrics_epoch{epoch+1}.h5'
                checkpoint_metrics_df.to_hdf(checkpoint_metrics_path, key='/data')

        train_loss_trackers['total'].reset_states()
        train_loss_trackers['reg'].reset_states()
        valid_loss_trackers['total'].reset_states()
        valid_loss_trackers['reg'].reset_states()
        for ii in range(n_outputs):
            train_loss_trackers['nll'][ii].reset_states()
            train_loss_trackers['evi'][ii].reset_states()
            valid_loss_trackers['nll'][ii].reset_states()
            valid_loss_trackers['evi'][ii].reset_states()
            train_performance_trackers['adjr2'][ii].reset_states()
            train_performance_trackers['mae'][ii].reset_states()
            train_performance_trackers['mse'][ii].reset_states()
            valid_performance_trackers['adjr2'][ii].reset_states()
            valid_performance_trackers['mae'][ii].reset_states()
            valid_performance_trackers['mse'][ii].reset_states()

        # Exit training loop early if requested by early stopping
        if stop_requested:
            break

    if stop_requested:
        logger.info(f'Early training loop exit triggered at epoch {epoch + 1}!')
    else:
        logger.info(f'Training loop exited at max epoch {epoch + 1}')

    last_index_to_keep = -n_no_improve if n_no_improve > 0 and stop_requested else None
    metrics_dict = {
        'train_total': total_train_list[:last_index_to_keep],
        'valid_total': total_valid_list[:last_index_to_keep],
        'train_reg': reg_train_list[:last_index_to_keep],
        'train_r2': r2_train_list[:last_index_to_keep],
        'train_mse': mse_train_list[:last_index_to_keep],
        'train_mae': mae_train_list[:last_index_to_keep],
        'train_nll': nll_train_list[:last_index_to_keep],
        'train_evi': evi_train_list[:last_index_to_keep],
        'valid_reg': reg_valid_list[:last_index_to_keep],
        'valid_r2': r2_valid_list[:last_index_to_keep],
        'valid_mse': mse_valid_list[:last_index_to_keep],
        'valid_mae': mae_valid_list[:last_index_to_keep],
        'valid_nll': nll_valid_list[:last_index_to_keep],
        'valid_evi': evi_valid_list[:last_index_to_keep],
    }

    return best_model, metrics_dict


def launch_tensorflow_pipeline_evidential(
    data,
    input_vars,
    output_vars,
    input_outlier_limit=None,
    output_outlier_limit=None,
    validation_fraction=0.1,
    test_fraction=0.1,
    data_split_file=None,
    max_epoch=100000,
    batch_size=None,
    early_stopping=50,
    minimum_performance=None,
    shuffle_seed=None,
    generalized_widths=None,
    specialized_depths=None,
    specialized_widths=None,
    l1_regularization=0.2,
    l2_regularization=0.8,
    relative_regularization=0.1,
    likelihood_weights=None,
    evidential_weights=None,
    regularization_weights=0.01,
    learning_rate=0.001,
    decay_rate=0.95,
    decay_epoch=10,
    log_file=None,
    checkpoint_freq=0,
    checkpoint_dir=None,
    save_initial_model=False,
    training_device=default_device,
    verbosity=0
):

    settings = {
        'input_outlier_limit': input_outlier_limit,
        'output_outlier_limit': output_outlier_limit,
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction,
        'data_split_file': data_split_file,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'minimum_performance': minimum_performance,
        'shuffle_seed': shuffle_seed,
        'generalized_widths': generalized_widths,
        'specialized_depths': specialized_depths,
        'specialized_widths': specialized_widths,
        'l1_regularization': l1_regularization,
        'l2_regularization': l2_regularization,
        'relative_regularization': relative_regularization,
        'likelihood_weights': likelihood_weights,
        'evidential_weights': evidential_weights,
        'regularization_weights': regularization_weights,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'decay_epoch': decay_epoch,
        'log_file': log_file,
        'checkpoint_freq': checkpoint_freq,
        'checkpoint_dir': checkpoint_dir,
        'save_initial_model': save_initial_model,
        'training_device': training_device,
    }

    if verbosity <= 4:
        set_tf_logging_level(logging.ERROR)

    if training_device not in ['cuda', 'gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        tf.config.set_visible_devices([], 'GPU')

    if verbosity >= 2:
        tf.config.run_functions_eagerly(True)

    lpath = Path(log_file) if isinstance(log_file, (str, Path)) else None
    if lpath is not None:
        setup_logging(logger, lpath, verbosity=verbosity)
    if verbosity >= 1:
        print_settings(logger, settings, 'Evidential model and training settings:')

    # Set up the required data sets
    start_preprocess = time.perf_counter()
    device_name, n_devices = get_device_info(training_device)
    if n_devices <= 0:
        raise RuntimeError(f'Requested device type, {training_device}, is not available on this system!')
    set_device_parallelism(n_devices)
    logger.info(f'Device type: {device_name}')
    logger.info(f'Number of devices: {n_devices}')
    device_list = [f'{device.name}'.replace('physical_device:', '') for device in tf.config.get_visible_devices(device_name)]
    strategy = tf.distribute.MirroredStrategy(devices=device_list)
    spath = Path(data_split_file) if isinstance(data_split_file, (str, Path)) else None
    features, targets = preprocess_data(
        data,
        input_vars,
        output_vars,
        validation_fraction,
        test_fraction,
        data_split_savepath=spath,
        seed=shuffle_seed,
        trim_feature_outliers=input_outlier_limit,
        trim_target_outliers=output_outlier_limit,
        logger=logger,
        verbosity=verbosity
    )
    if verbosity >= 2:
        logger.debug(f'  Input scaling mean: {features["scaler"].mean_}')
        logger.debug(f'  Input scaling std: {features["scaler"].scale_}')
        logger.debug(f'  Output scaling mean: {targets["scaler"].mean_}')
        logger.debug(f'  Output scaling std: {targets["scaler"].scale_}')
    end_preprocess = time.perf_counter()

    logger.info(f'Pre-processing completed! Elapsed time: {(end_preprocess - start_preprocess):.4f} s')

    # Set up the Evidential BNN model
    start_setup = time.perf_counter()
    n_inputs = features['train'].shape[-1]
    n_outputs = targets['train'].shape[-1]
    n_commons = len(generalized_widths) if isinstance(generalized_widths, (list, tuple)) else 0
    common_nodes = list(generalized_widths) if n_commons > 0 else None
    special_nodes = None
    if isinstance(specialized_depths, (list, tuple)) and len(specialized_depths) > 0 and isinstance(specialized_widths, (list, tuple)) and len(specialized_widths) > 0:
        special_nodes = []
        ll = 0
        for jj in range(len(specialized_depths)):
            output_special_nodes = []
            for kk in range(specialized_depths[jj]):
                special_layer_width = specialized_widths[ll] if ll < len(specialized_widths) else specialized_widths[-1]
                output_special_nodes.append(specialized_widths[ll])
                ll += 1
            special_nodes.append(output_special_nodes)   # List of lists
    with strategy.scope():
        model = create_regressor_model(
            n_input=n_inputs,
            n_output=n_outputs,
            n_common=n_commons,
            common_nodes=common_nodes,
            special_nodes=special_nodes,
            regpar_l1=l1_regularization,
            regpar_l2=l2_regularization,
            relative_regpar=relative_regularization,
            style='evidential',
            verbosity=verbosity
        )

    # Set up the user-defined loss term weights, default behaviour included if input is None
    nll_weights = [1.0] * n_outputs
    for ii in range(n_outputs):
        if isinstance(likelihood_weights, list):
            nll_weights[ii] = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
    evi_weights = [1.0] * n_outputs
    for ii in range(n_outputs):
        if isinstance(evidential_weights, list):
            evi_weights[ii] = evidential_weights[ii] if ii < len(evidential_weights) else evidential_weights[-1]

    # Create custom loss function, weights converted into tensor objects internally
    with strategy.scope():
        loss_function = create_regressor_loss_function(
            n_outputs,
            style='evidential',
            nll_weights=nll_weights,
            evi_weights=evi_weights,
            verbosity=verbosity
        )

    train_length = features['train'].shape[0]
    steps_per_epoch = int(np.ceil(train_length / batch_size)) if isinstance(batch_size, int) else 1
    decay_steps = steps_per_epoch * decay_epoch
    with strategy.scope():
        optimizer, scheduler = create_scheduled_adam_optimizer(
            model=model,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    end_setup = time.perf_counter()

    logger.info(f'Setup completed! Elapsed time: {(end_setup - start_setup):.4f} s')

    # Perform the training loop
    start_train = time.perf_counter()
    checkpoint_path = Path(checkpoint_dir) if isinstance(checkpoint_dir, (str, Path)) else None
    if checkpoint_path is not None and not checkpoint_path.is_dir():
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        else:
            logger.warning(f'Requested checkpoint directory, {checkpoint_path}, exists and is not a directory. Checkpointing will be skipped!')
            checkpoint_path = None
    if save_initial_model:
        if checkpoint_path is not None and checkpoint_path.is_dir():
            initpath = checkpoint_path / 'checkpoint_model_initial.keras'
            initial_model = tf.keras.models.clone_model(model)
            initial_model.set_weights(model.get_weights())
            if 'scaler' in features and features['scaler'] is not None and 'scaler' in targets and targets['scaler'] is not None:
                initial_model = wrap_regressor_model(initial_model, features['scaler'], targets['scaler'])
            save_model(initial_model, initpath)
        else:
            logger.warning(f'Requested initial model save cannot be made due to invalid checkpoint directory, {checkpoint_path}. Initial save will be skipped!')
            checkpoint_path = None
    best_model, metrics = train_tensorflow_evidential(
        strategy,
        model,
        optimizer,
        features['train'],
        targets['train'],
        features['validation'],
        targets['validation'],
        loss_function,
        regularization_weights,
        max_epoch,
        batch_size=batch_size,
        patience=early_stopping,
        r2_minimums=minimum_performance,
        checkpoint_freq=checkpoint_freq,
        checkpoint_path=checkpoint_path,
        features_scaler=features['scaler'],
        targets_scaler=targets['scaler'],
        verbosity=verbosity
    )
    end_train = time.perf_counter()

    logger.info(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

    # Configure the trained model and training metrics for saving
    start_out = time.perf_counter()
    metrics_dict = {}
    for key, val in metrics.items():
        if key.endswith('total') or key.endswith('reg'):
            metric = np.array(val)
            metrics_dict[f'{key}'] = metric.flatten()
        else:
            metric = np.atleast_2d(val)
            for ii in range(n_outputs):
                metrics_dict[f'{key}{ii}'] = metric[:, ii].flatten()
    metrics_df = pd.DataFrame(data=metrics_dict)
    wrapped_model = wrap_regressor_model(best_model, features['scaler'], targets['scaler'])
    end_out = time.perf_counter()

    logger.info(f'Output configuration completed! Elapsed time: {(end_out - start_out):.4f} s')

    if verbosity >= 2:
        inputs = tf.zeros([1, best_model.n_inputs], dtype=default_dtype)
        outputs = model(inputs)
        logger.debug(f'  Sample output at origin:')
        logger.debug(f'{outputs}')

    return wrapped_model, metrics_df


def main():

    args = parse_inputs()

    ipath = Path(args.data_file)
    mpath = Path(args.metrics_file)
    npath = Path(args.network_file)
    device = default_device if not args.disable_gpu else 'cpu'

    if not ipath.is_file():
        raise IOError(f'Could not find input data file: {ipath}')

    lpath = Path(args.log_file) if isinstance(args.log_file, str) else None
    setup_logging(logger, lpath, args.verbosity)
    logger.info(f'Starting Evidential BNN training script...')
    if args.verbosity >= 1:
        print_settings(logger, vars(args), 'Evidential training pipeline CLI settings:')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_tensorflow_pipeline_evidential(
        data=data,
        input_vars=args.input_var,
        output_vars=args.output_var,
        input_outlier_limit=args.input_trim,
        output_outlier_limit=args.output_trim,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        data_split_file=args.data_split_file,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        minimum_performance=args.minimum_performance,
        shuffle_seed=args.shuffle_seed,
        generalized_widths=args.generalized_node,
        specialized_depths=args.specialized_layer,
        specialized_widths=args.specialized_node,
        l1_regularization=args.l1_reg_general,
        l2_regularization=args.l2_reg_general,
        relative_regularization=args.rel_reg_special,
        likelihood_weights=args.nll_weight,
        evidential_weights=args.evi_weight,
        regularization_weights=args.reg_weight,
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_epoch=args.decay_epoch,
        log_file=lpath,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        save_initial_model=args.save_initial,
        training_device=device,
        verbosity=args.verbosity
    )

    metrics_dict.to_hdf(mpath, key='/data')
    logger.info(f' Metrics saved in {mpath}')

    save_model(trained_model, npath)
    logger.info(f' Network saved in {npath}')

    end_pipeline = time.perf_counter()

    logger.info(f'Pipeline completed! Total time: {(end_pipeline - start_pipeline):.4f} s')

    logger.info(f'Script completed!')


if __name__ == "__main__":
    main()

