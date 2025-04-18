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
    create_classifier_model,
    create_classifier_loss_function,
    wrap_classifier_model,
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
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--validation_data_file', metavar='path', type=str, default=None, help='Optional path of HDF5 file containing an independent validation set, overwrites any splitting of training data set')
    parser.add_argument('--data_split_file', metavar='path', type=str, default=None, help='Optional path and name of output HDF5 file of training, validation, and test dataset split indices')
    parser.add_argument('--max_epoch', metavar='n', type=int, default=100000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=50, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--minimum_performance', metavar='val', type=float, nargs='*', default=None, help='Set minimum value in F-beta=1 per output before early stopping is activated')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--generalized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the generalized hidden layers')
    parser.add_argument('--specialized_layer', metavar='n', type=int, nargs='*', default=None, help='Number of specialized hidden layers, given for each output')
    parser.add_argument('--specialized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layers, sequential per output stack')
    parser.add_argument('--spec_norm_general', metavar='wgt', type=float, default=0.9, help='Spectral normalization parameter used in the generalized hidden layers')
    parser.add_argument('--rel_norm_special', metavar='wgt', type=float, default=1.0, help='Relative spectral normalization used in the specialized hidden layers compared to the generalized layers')
    parser.add_argument('--entropy_weight', metavar='wgt', type=float, default=1.0, help='Weight to apply to the cross-entropy loss term')
    parser.add_argument('--reg_weight', metavar='wgt', type=float, default=1.0, help='Weight to apply to regularization loss term (not applicable here)')
    parser.add_argument('--n_class', metavar='n', type=int, default=1, help='Total number of possible classes present in classification target data')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.95, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epoch', metavar='n', type=float, default=10, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to output log file where script related print outs will be stored')
    parser.add_argument('--checkpoint_freq', metavar='n', type=int, default=0, help='Number of epochs between saves of model checkpoint')
    parser.add_argument('--checkpoint_dir', metavar='path', type=str, default=None, help='Optional path to directory where checkpoints will be saved')
    parser.add_argument('--save_initial', default=False, action='store_true', help='Toggle on saving of initialized model before any training, for debugging')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


@tf.function
def train_tensorflow_sngp_step(
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
    #n_classes = model.n_outputs

    replica_context = tf.distribute.get_replica_context()
    if replica_context is not None:
        batch_size = tf.cast(tf.reduce_sum(replica_context.all_gather(tf.stack([tf.shape(feature_batch)], axis=0), axis=0), axis=0)[0], dtype=default_dtype)
    else:
        batch_size = tf.cast(feature_shape[0], dtype=default_dtype)

    # Set up training targets into a single large tensor
    batch_loss_targets = target_batch

    with tf.GradientTape() as tape:

        # For mean data inputs, e.g. training data
        outputs = model(feature_batch, training=training)

        if training and tf.executing_eagerly() and verbosity >= 4:
            for ii in range(n_outputs):
                logger.debug(f'     logit {ii}: {outputs[0, 0, ii]}')
                logger.debug(f'     variance {ii}: {outputs[0, 1, ii]}')

        batch_loss_predictions = tf.squeeze(tf.gather(outputs, indices=[0], axis=1), axis=1)

        # Compute total loss to be used in adjusting weights and biases
        step_total_loss = loss_function(batch_loss_targets, batch_loss_predictions)
        adjusted_step_total_loss = tf.math.divide(step_total_loss, batch_size)
        step_entropy_loss = step_total_loss

    # Apply back-propagation
    if training:
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(adjusted_step_total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

    return (
        tf.reshape(step_total_loss, shape=(-1, 1)),
        tf.reshape(step_entropy_loss, shape=(-1, n_outputs))
    )


@tf.function
def distributed_train_tensorflow_sngp_step(
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

    replica_total_loss, replica_entropy_loss = strategy.run(
        train_tensorflow_sngp_step,
        args=(model, optimizer, loss_function, feature_batch, target_batch, reg_weight, dataset_size, training, verbosity)
    )
    return (
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_total_loss, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_entropy_loss, axis=0)
    )


@tf.function
def train_tensorflow_sngp_epoch(
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
    step_entropy_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'entropy_loss_array')

    # Custom model function resets the covariance matrix, critical for proper training of SNGP architecture
    if hasattr(model, 'pre_epoch_processing'):
        model.pre_epoch_processing()

    # Training loop through minibatches - each loop pass is one step
    nn = 0
    for feature_batch, target_batch in dataloader:

        # Evaluate training step on batch using distribution strategy
        step_total_loss, step_entropy_loss = distributed_train_tensorflow_sngp_step(
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
        step_entropy_losses = step_entropy_losses.write(fill_index, tf.reshape(step_entropy_loss, shape=[-1, n_outputs]))

        nn += 1

    epoch_total_loss = tf.reduce_sum(step_total_losses.concat(), axis=0)
    epoch_entropy_loss = tf.reduce_sum(step_entropy_losses.concat(), axis=0)

    return (
        epoch_total_loss,
        epoch_entropy_loss
    )


def meter_tensorflow_sngp_epoch(
    model,
    inputs,
    targets,
    losses,
    loss_trackers={},
    performance_trackers={},
    dataset_length=None,
    section_length=None,
    beta=1.0,
    verbosity=0
):

    n_inputs = model.n_inputs
    n_outputs = model.n_outputs
    dataset_size = inputs.shape[0] if dataset_length is None else dataset_length
    section_max = dataset_size if section_length is None else section_length
    total_loss, entropy_loss = losses

    outputs = model(inputs, training=False)
    means = tf.squeeze(tf.gather(outputs, indices=[0], axis=1), axis=1)
    #roc_thresholds = torch.constant(np.linspace(0.0, 1.0, 101)[1:-1], dtype=outputs.dtype)

    section_outputs = []
    data_section_labels = np.arange(inputs.shape[0]) // section_max   # Floor division
    section_labels, section_indices = np.unique(data_section_labels, return_inverse=True)
    for nn in range(len(section_labels)):
        section_mask = (section_indices == nn)
        section_outputs.append(model(inputs[section_mask, :], training=False))
    outputs = np.concatenate(section_outputs, axis=0)
    means = tf.squeeze(tf.gather(outputs, indices=[0], axis=1), axis=1)
    variances = tf.squeeze(tf.gather(outputs, indices=[1], axis=1), axis=1)
    probs = tf.math.sigmoid(means / tf.sqrt(1.0 + (tf.math.acos(tf.constant([1.0], dtype=default_dtype)) / 8.0) * variances))

    entropy_metric = [np.nan] * n_outputs
    #f1_metric = [np.nan] * n_outputs
    auc_metric = [np.nan] * n_outputs
    tp_metric = [np.nan] * n_outputs
    tn_metric = [np.nan] * n_outputs
    fp_metric = [np.nan] * n_outputs
    fn_metric = [np.nan] * n_outputs
    fbeta_metric = [np.nan] * n_outputs
    threshold_metric = [np.nan] * n_outputs

    loss_trackers['total'].update_state(total_loss / dataset_size)
    total_metric = loss_trackers['total'].result()

    for ii in range(n_outputs):

        metric_targets = np.atleast_2d(targets[:, ii]).T
        metric_results = np.atleast_2d(probs[:, ii].numpy()).T

        loss_trackers['entropy'][ii].update_state(entropy_loss / dataset_size)
        entropy_metric[ii] = loss_trackers['entropy'][ii].result()

        #performance_trackers['f1'][ii].update_state(metric_targets, metric_results)
        #f1_metric[ii] = performance_trackers['f1'][ii].result()

        performance_trackers['auc'][ii].update_state(metric_targets, metric_results)
        auc_metric[ii] = performance_trackers['auc'][ii].result()

        performance_trackers['tp'][ii].update_state(metric_targets, metric_results)
        performance_trackers['tn'][ii].update_state(metric_targets, metric_results)
        performance_trackers['fp'][ii].update_state(metric_targets, metric_results)
        performance_trackers['fn'][ii].update_state(metric_targets, metric_results)

        tp_curve = performance_trackers['tp'][ii].result()
        tn_curve = performance_trackers['tn'][ii].result()
        fp_curve = performance_trackers['fp'][ii].result()
        fn_curve = performance_trackers['fn'][ii].result()
        roc_thresholds = tf.constant(performance_trackers['tp'][ii].init_thresholds, dtype=tp_curve.dtype)
        b2 = tf.constant(beta ** 2.0, dtype=tp_curve.dtype)
        ones = tf.constant(1.0, dtype=tp_curve.dtype)
        tp_factor = tf.multiply(tf.add(ones, b2), tp_curve)
        fn_factor = tf.multiply(b2, fn_curve)
        fb_curve = tf.divide(tp_factor, tf.add(tf.add(tp_factor, fn_factor), fp_curve))
        opt_index = tf.math.argmax(fb_curve)

        tp_metric[ii] = tf.gather(tp_curve, indices=opt_index, axis=0)
        tn_metric[ii] = tf.gather(tn_curve, indices=opt_index, axis=0)
        fp_metric[ii] = tf.gather(fp_curve, indices=opt_index, axis=0)
        fn_metric[ii] = tf.gather(fn_curve, indices=opt_index, axis=0)
        fbeta_metric[ii] = tf.gather(fb_curve, indices=opt_index, axis=0)
        threshold_metric[ii] = tf.gather(roc_thresholds, indices=opt_index, axis=0)

    entropy_metric = tf.stack(entropy_metric, axis=0)
    auc_metric = tf.stack(auc_metric, axis=0)
    tp_metric = tf.stack(tp_metric, axis=0)
    tn_metric = tf.stack(tn_metric, axis=0)
    fp_metric = tf.stack(fp_metric, axis=0)
    fn_metric = tf.stack(fn_metric, axis=0)
    fbeta_metric = tf.stack(fbeta_metric, axis=0)
    threshold_metric = tf.stack(threshold_metric, axis=0)

    return (
        tf.stack([total_metric], axis=0),
        tf.stack([entropy_metric], axis=0),
        tf.stack([auc_metric], axis=0),
        tf.stack([tp_metric], axis=0),
        tf.stack([tn_metric], axis=0),
        tf.stack([fp_metric], axis=0),
        tf.stack([fn_metric], axis=0),
        tf.stack([fbeta_metric], axis=0),
        tf.stack([threshold_metric], axis=0)
    )


def distributed_meter_tensorflow_sngp_epoch(
    strategy,
    model,
    inputs,
    targets,
    losses,
    loss_trackers={},
    performance_trackers={},
    dataset_length=None,
    section_length=None,
    beta=1.0,
    verbosity=0
):

    replica_total_metric, replica_entropy_metric, replica_auc_metric, replica_tp_metric, replica_tn_metric, replica_fp_metric, replica_fn_metric, replica_fbeta_metric, replica_threshold_metric = strategy.run(
        meter_tensorflow_sngp_epoch,
        args=(model, inputs, targets, losses, loss_trackers, performance_trackers, dataset_length, section_length, beta, verbosity)
    )

    return (
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_total_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.SUM, replica_entropy_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_auc_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_tp_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_tn_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_fp_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_fn_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_fbeta_metric, axis=0),
        strategy.reduce(tf.distribute.ReduceOp.MEAN, replica_threshold_metric, axis=0)
    )


def train_tensorflow_sngp(
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
    f1_minimums=None,
    seed=None,
    checkpoint_freq=0,
    checkpoint_path=None,
    features_scaler=None,
    targets_names=None,
    verbosity=0
):

    n_inputs = features_train.shape[-1]
    n_outputs = targets_train.shape[-1]
    train_length = features_train.shape[0]
    valid_length = features_valid.shape[0]
    n_no_improve = 0
    improve_tol = 0.0
    section_max = 1000
    idx_def = 50
    roc_thresholds = np.linspace(0.0, 1.0, 101).tolist()[1:-1]
    beta = 1.0
    optimal_class_ratio = 0.5
    f1_thresholds = None
    if isinstance(f1_minimums, (list, tuple, np.ndarray)):
        f1_thresholds = [-1.0] * n_outputs
        for ii in range(n_outputs):
            f1_thresholds[ii] = float(f1_minimums[ii]) if ii < len(f1_minimums) else -1.0
    #multi_label = (n_outputs > 1)   # TODO: Better handling of multi-class outputs

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
            'entropy': [],
        }
        for ii in range(n_outputs):
            train_loss_trackers['entropy'].append(tf.keras.metrics.Sum(name=f'train_entropy{ii}', dtype=default_dtype))

        train_performance_trackers = {
            #'f1': [],
            'auc': [],
            'tp': [],
            'tn': [],
            'fp': [],
            'fn': [],
        }
        for ii in range(n_outputs):
            train_performance_trackers['auc'].append(tf.keras.metrics.AUC(num_thresholds=101, name=f'train_auc{ii}', dtype=default_dtype))
            train_performance_trackers['tp'].append(tf.keras.metrics.TruePositives(thresholds=roc_thresholds, name=f'train_tp{ii}', dtype=default_dtype))
            train_performance_trackers['tn'].append(tf.keras.metrics.TrueNegatives(thresholds=roc_thresholds, name=f'train_tn{ii}', dtype=default_dtype))
            train_performance_trackers['fp'].append(tf.keras.metrics.FalsePositives(thresholds=roc_thresholds, name=f'train_fp{ii}', dtype=default_dtype))
            train_performance_trackers['fn'].append(tf.keras.metrics.FalseNegatives(thresholds=roc_thresholds, name=f'train_fn{ii}', dtype=default_dtype))

        # Create validation tracker objects to facilitate external analysis of pipeline
        valid_loss_trackers = {
            'total': tf.keras.metrics.Sum(name=f'valid_total', dtype=default_dtype),
            'entropy': [],
        }
        for ii in range(n_outputs):
            valid_loss_trackers['entropy'].append(tf.keras.metrics.Sum(name=f'valid_entropy{ii}', dtype=default_dtype))

        valid_performance_trackers = {
            #'f1': [],
            'auc': [],
            'tp': [],
            'tn': [],
            'fp': [],
            'fn': [],
        }
        for ii in range(n_outputs):
            valid_performance_trackers['auc'].append(tf.keras.metrics.AUC(num_thresholds=101, name=f'valid_auc{ii}', dtype=default_dtype))
            valid_performance_trackers['tp'].append(tf.keras.metrics.TruePositives(thresholds=roc_thresholds, name=f'valid_tp{ii}', dtype=default_dtype))
            valid_performance_trackers['tn'].append(tf.keras.metrics.TrueNegatives(thresholds=roc_thresholds, name=f'valid_tn{ii}', dtype=default_dtype))
            valid_performance_trackers['fp'].append(tf.keras.metrics.FalsePositives(thresholds=roc_thresholds, name=f'valid_fp{ii}', dtype=default_dtype))
            valid_performance_trackers['fn'].append(tf.keras.metrics.FalseNegatives(thresholds=roc_thresholds, name=f'valid_fn{ii}', dtype=default_dtype))

    # Output metrics containers
    total_train_list = []
    entropy_train_list = []
    #f1_train_list = []
    auc_train_list = []
    tp_train_list = []
    tn_train_list = []
    fp_train_list = []
    fn_train_list = []
    fb_train_list = []
    thr_train_list = []
    total_valid_list = []
    entropy_valid_list = []
    #f1_valid_list = []
    auc_valid_list = []
    tp_valid_list = []
    tn_valid_list = []
    fp_valid_list = []
    fn_valid_list = []
    fb_valid_list = []
    thr_valid_list = []

    # Output container for the best trained model
    best_validation_loss = None
    best_model = tf.keras.models.clone_model(model)
    best_model.set_weights(model.get_weights())

    # Training loop
    stop_requested = False
    thresholds_surpassed = [False] * n_outputs if isinstance(f1_thresholds, list) else [True] * n_outputs
    current_thresholds_surpassed = [False] * n_outputs if isinstance(f1_thresholds, list) else [True] * n_outputs
    for epoch in range(max_epochs):

        # Training routine described in here
        train_losses = train_tensorflow_sngp_epoch(
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
        train_metrics = distributed_meter_tensorflow_sngp_epoch(
            strategy,
            model,
            train_data[0],
            train_data[1],
            train_losses,
            loss_trackers=train_loss_trackers,
            performance_trackers=train_performance_trackers,
            dataset_length=train_length,
            section_length=section_max,
            beta=beta,
            verbosity=verbosity
        )
        train_total, train_entropy, train_auc, train_tp, train_tn, train_fp, train_fn, train_fb, train_thr = train_metrics

        total_train_list.append(train_total.numpy().tolist())
        entropy_train_list.append(train_entropy.numpy().tolist())
        #f1_train_list.append(train_f1.numpy().tolist())
        auc_train_list.append(train_auc.numpy().tolist())
        tp_train_list.append(train_tp.numpy().tolist())
        tn_train_list.append(train_tn.numpy().tolist())
        fp_train_list.append(train_fp.numpy().tolist())
        fn_train_list.append(train_fn.numpy().tolist())
        fb_train_list.append(train_fb.numpy().tolist())
        thr_train_list.append(train_thr.numpy().tolist())

        # Reuse training routine to evaluate validation data
        valid_losses = train_tensorflow_sngp_epoch(
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
        valid_metrics = distributed_meter_tensorflow_sngp_epoch(
            strategy,
            model,
            valid_data[0],
            valid_data[1],
            valid_losses,
            loss_trackers=valid_loss_trackers,
            performance_trackers=valid_performance_trackers,
            dataset_length=valid_length,
            section_length=section_max,
            beta=beta,
            verbosity=verbosity
        )
        valid_total, valid_entropy, valid_auc, valid_tp, valid_tn, valid_fp, valid_fn, valid_fb, valid_thr = valid_metrics

        total_valid_list.append(valid_total.numpy().tolist())
        entropy_valid_list.append(valid_entropy.numpy().tolist())
        #f1_valid_list.append(valid_f1.numpy().tolist())
        auc_valid_list.append(valid_auc.numpy().tolist())
        tp_valid_list.append(valid_tp.numpy().tolist())
        tn_valid_list.append(valid_tn.numpy().tolist())
        fp_valid_list.append(valid_fp.numpy().tolist())
        fn_valid_list.append(valid_fn.numpy().tolist())
        fb_valid_list.append(valid_fb.numpy().tolist())
        thr_valid_list.append(valid_thr.numpy().tolist())

        # Set optimal thresholds using ROC analysis
        model.set_thresholds([float(val) for val in thr_train_list[-1]])

        # Enable early stopping routine if minimum performance threshold is met
        if isinstance(f1_thresholds, list) and not all(current_thresholds_surpassed):
            individual_minimum_flag = True if all(thresholds_surpassed) else False
            if not np.all(np.isfinite(fb_valid_list[-1])):
                for ii in range(n_outputs):
                    thresholds_surpassed[ii] = True
                    current_thresholds_surpassed[ii] = True
                logger.warning(f'An F-beta=1 value of NaN was detected, enabling early stopping to prevent large computational waste...')
            else:
                for ii in range(n_outputs):
                    if fb_valid_list[-1][ii] >= f1_thresholds[ii]:
                        if not thresholds_surpassed[ii] and f1_thresholds[ii] >= 0.0:
                            logger.info(f'Requested minimum performance on Output {ii} of {f1_thresholds[ii]:.5f} exceeded at epoch {epoch + 1}')
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
            logger.info(f' {epoch_str} Train -- total_train = {total_train_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.info(f'  -> Output {ii}: fb = {fb_train_list[-1][ii]:.3f}, auc = {auc_train_list[-1][ii]:.3f}, threshold = {thr_train_list[-1][ii]:.2f}, entropy = {entropy_train_list[-1][ii]:.3f}')
            logger.info(f' {epoch_str} Valid -- total_valid = {total_valid_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.info(f'  -> Output {ii}: fb = {fb_valid_list[-1][ii]:.3f}, auc = {auc_valid_list[-1][ii]:.3f}, threshold = {thr_valid_list[-1][ii]:.2f}, entropy = {entropy_valid_list[-1][ii]:.3f}')

        # Model Checkpoint
        # ------------------------------------------------
        if checkpoint_path is not None and checkpoint_freq > 0:
            if (epoch + 1) % checkpoint_freq == 0:
                check_path = checkpoint_path / f'checkpoint_model_epoch{epoch+1}.keras'
                checkpoint_model = tf.keras.models.clone_model(model)
                checkpoint_model.set_weights(model.get_weights())
                if features_scaler is not None and targets_names:
                    checkpoint_model = wrap_classifier_model(checkpoint_model, features_scaler, targets_names)
                save_model(checkpoint_model, check_path)

                checkpoint_metrics_dict = {
                    'train_total': total_train_list,
                    'valid_total': total_valid_list,
                    'train_auc': auc_train_list,
                    'train_tp': tp_train_list,
                    'train_tn': tn_train_list,
                    'train_fp': fp_train_list,
                    'train_fn': fn_train_list,
                    'train_fbeta': fb_train_list,
                    'train_threshold': thr_train_list,
                    'train_entropy': entropy_train_list,
                    'valid_auc': auc_valid_list,
                    'valid_tp': tp_valid_list,
                    'valid_tn': tn_valid_list,
                    'valid_fp': fp_valid_list,
                    'valid_fn': fn_valid_list,
                    'valid_fbeta': fb_valid_list,
                    'valid_threshold': thr_valid_list,
                    'valid_entropy': entropy_valid_list,
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
        valid_loss_trackers['total'].reset_states()
        for ii in range(n_outputs):
            train_loss_trackers['entropy'][ii].reset_states()
            #train_performance_trackers['f1'][ii].reset_states()
            train_performance_trackers['auc'][ii].reset_states()
            train_performance_trackers['tp'][ii].reset_states()
            train_performance_trackers['tn'][ii].reset_states()
            train_performance_trackers['fp'][ii].reset_states()
            train_performance_trackers['fn'][ii].reset_states()
            valid_loss_trackers['entropy'][ii].reset_states()
            #valid_performance_trackers['f1'][ii].reset_states()
            valid_performance_trackers['auc'][ii].reset_states()
            valid_performance_trackers['tp'][ii].reset_states()
            valid_performance_trackers['tn'][ii].reset_states()
            valid_performance_trackers['fp'][ii].reset_states()
            valid_performance_trackers['fn'][ii].reset_states()

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
        'train_auc': auc_train_list[:last_index_to_keep],
        #'train_f1': f1_train_list[:last_index_to_keep],
        'train_tp': tp_train_list[:last_index_to_keep],
        'train_tn': tn_train_list[:last_index_to_keep],
        'train_fp': fp_train_list[:last_index_to_keep],
        'train_fn': fn_train_list[:last_index_to_keep],
        'train_fbeta': fb_train_list[:last_index_to_keep],
        'train_threshold': thr_train_list[:last_index_to_keep],
        'train_entropy': entropy_train_list[:last_index_to_keep],
        'valid_auc': auc_valid_list[:last_index_to_keep],
        #'valid_f1': f1_valid_list[:last_index_to_keep],
        'valid_tp': tp_valid_list[:last_index_to_keep],
        'valid_tn': tn_valid_list[:last_index_to_keep],
        'valid_fp': fp_valid_list[:last_index_to_keep],
        'valid_fn': fn_valid_list[:last_index_to_keep],
        'valid_fbeta': fb_valid_list[:last_index_to_keep],
        'valid_threshold': thr_valid_list[:last_index_to_keep],
        'valid_entropy': entropy_valid_list[:last_index_to_keep],
    }

    return best_model, metrics_dict


def launch_tensorflow_pipeline_sngp(
    data,
    input_vars,
    output_vars,
    input_outlier_limit=None,
    validation_fraction=0.1,
    test_fraction=0.1,
    validation_data_file=None,
    data_split_file=None,
    max_epoch=100000,
    batch_size=None,
    early_stopping=50,
    minimum_performance=None,
    shuffle_seed=None,
    generalized_widths=None,
    specialized_depths=None,
    specialized_widths=None,
    spectral_normalization=0.9,
    relative_normalization=1.0,
    entropy_weights=1.0,
    regularization_weights=1.0,
    total_classes=1,
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
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction,
        'validation_data_file': validation_data_file,
        'data_split_file': data_split_file,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'minimum_performance': minimum_performance,
        'shuffle_seed': shuffle_seed,
        'generalized_widths': generalized_widths,
        'specialized_depths': specialized_depths,
        'specialized_widths': specialized_widths,
        'spectral_normalization': spectral_normalization,
        'relative_normalization': relative_normalization,
        'entropy_weights': entropy_weights,
        'regularization_weights': regularization_weights,
        'total_classes': total_classes,
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
        print_settings(logger, settings, 'SNGP model and training settings:')

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
    vpath = Path(validation_data_file) if isinstance(validation_data_file, (str, Path)) else None
    spath = Path(data_split_file) if isinstance(data_split_file, (str, Path)) else None
    features, targets = preprocess_data(
        data,
        input_vars,
        output_vars,
        validation_fraction,
        test_fraction,
        validation_loadpath=vpath,
        data_split_savepath=spath,
        seed=shuffle_seed,
        trim_feature_outliers=input_outlier_limit,
        trim_target_outliers=None,
        scale_features=True,
        scale_targets=False,
        logger=logger,
        verbosity=verbosity
    )
    if verbosity >= 2:
        logger.debug(f'  Input scaling mean: {features["scaler"].mean_}')
        logger.debug(f'  Input scaling std: {features["scaler"].scale_}')
    end_preprocess = time.perf_counter()

    logger.info(f'Pre-processing completed! Elapsed time: {(end_preprocess - start_preprocess):.4f} s')

    # Set up the SNGP BNN model
    start_setup = time.perf_counter()
    model_type = 'sngp'
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
        model = create_classifier_model(
            n_input=n_inputs,
            n_output=n_outputs,
            n_common=n_commons,
            common_nodes=common_nodes,
            special_nodes=special_nodes,
            spectral_norm=spectral_normalization,
            relative_norm=relative_normalization,
            style=model_type,
            verbosity=verbosity
        )

    # Create custom loss function, weights converted into tensor objects internally
    with strategy.scope():
        loss_function = create_classifier_loss_function(
            n_outputs,
            style=model_type,
            h_weights=entropy_weights,
            n_classes=total_classes,
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
            if 'scaler' in features and features['scaler'] is not None and output_vars is not None:
                initial_model = wrap_classifier_model(initial_model, features['scaler'], output_vars)
            save_model(initial_model, initpath)
        else:
            logger.warning(f'Requested initial model save cannot be made due to invalid checkpoint directory, {checkpoint_path}. Initial save will be skipped!')
            checkpoint_path = None
    best_model, metrics = train_tensorflow_sngp(
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
        f1_minimums=minimum_performance,
        checkpoint_freq=checkpoint_freq,
        checkpoint_path=checkpoint_path,
        features_scaler=features['scaler'],
        targets_names=output_vars,
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
    wrapped_model = wrap_classifier_model(best_model, features['scaler'], output_vars)
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
    logger.info(f'Starting SNGP BNN training script...')
    if args.verbosity >= 1:
        print_settings(logger, vars(args), 'SNGP training pipeline CLI settings:')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_tensorflow_pipeline_sngp(
        data=data,
        input_vars=args.input_var,
        output_vars=args.output_var,
        input_outlier_limit=args.input_trim,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        validation_data_file=args.validation_data_file,
        data_split_file=args.data_split_file,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        minimum_performance=args.minimum_performance,
        shuffle_seed=args.shuffle_seed,
        generalized_widths=args.generalized_node,
        specialized_depths=args.specialized_layer,
        specialized_widths=args.specialized_node,
        spectral_normalization=args.spec_norm_general,
        relative_normalization=args.rel_norm_special,
        entropy_weights=args.entropy_weight,
        regularization_weights=args.reg_weight,
        total_classes=args.n_class,
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

