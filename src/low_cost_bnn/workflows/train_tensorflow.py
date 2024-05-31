import sys
import argparse
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
#from tensorflow_probability import distributions as tfd
from ..utils.helpers import create_scaler, split
from ..utils.helpers_tensorflow import create_data_loader, create_scheduled_adam_optimizer, create_model, create_loss_function, wrap_model

logger = logging.getLogger("train_tensorflow")

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Input HDF5 file containing training data set')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of HDF5 file to store training metrics')
    parser.add_argument('--input_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--max_epoch', metavar='n', type=int, default=10000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=None, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--sample_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for OOD sampling')
    parser.add_argument('--generalized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the generalized hidden layers')
    parser.add_argument('--specialized_layer', metavar='n', type=int, nargs='*', default=None, help='Number of specialized hidden layers, given for each output')
    parser.add_argument('--specialized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layers, sequential per output stack')
    parser.add_argument('--ood_width', metavar='val', type=float, default=0.2, help='Normalized standard deviation of OOD sampling distribution')
    parser.add_argument('--epi_prior', metavar='val', type=float, nargs='*', default=None, help='Standard deviation of epistemic priors used to compute epistemic loss term')
    parser.add_argument('--alea_prior', metavar='val', type=float, nargs='*', default=None, help='Standard deviation of aleatoric priors used to compute aleatoric loss term')
    parser.add_argument('--nll_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the NLL loss term')
    parser.add_argument('--epi_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to epistemic loss term')
    parser.add_argument('--alea_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to aleatoric loss term')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.98, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epoch', metavar='n', type=float, default=20, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to log file where script related print outs will be stored')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def print_settings(settings):
    logger.debug(f'Input settings print-out requested...')
    for key, val in settings.items():
       logger.debug(f'  {key}: {val}')


def setup_logging(log_path=None, verbosity=0):

    formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    if isinstance(log_path, Path):
        log = logging.FileHandler(str(log_path), mode='w')
        log.setLevel(logging.DEBUG)
        log.setFormatter(formatter)
        logger.addHandler(log)

    else:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(logging.DEBUG)
        stream.setFormatter(formatter)
        logger.addHandler(stream)


def preprocess_data(
    data,
    feature_vars,
    target_vars,
    validation_fraction,
    test_fraction,
    shuffle=True,
    seed=None,
    verbosity=0
):

    ml_vars = []
    ml_vars.extend(feature_vars)
    ml_vars.extend(target_vars)
    ml_data = data.loc[:, ml_vars].astype(np.float32)

    feature_scaler = create_scaler(ml_data.loc[:, feature_vars])
    target_scaler = create_scaler(ml_data.loc[:, target_vars])
    if verbosity >= 2:
        logger.debug(f'  Input scaling mean: {feature_scaler.mean_}')
        logger.debug(f'  Input scaling std: {feature_scaler.scale_}')
        logger.debug(f'  Output scaling mean: {target_scaler.mean_}')
        logger.debug(f'  Output scaling std: {target_scaler.scale_}')

    first_split = validation_fraction + test_fraction
    second_split = test_fraction / first_split
    train_data, split_data = split(ml_data, first_split, shuffle=shuffle, seed=seed)
    val_data, test_data = split(split_data, second_split, shuffle=shuffle, seed=seed)

    feature_train = feature_scaler.transform(train_data.loc[:, feature_vars])
    feature_val = feature_scaler.transform(val_data.loc[:, feature_vars])
    feature_test = feature_scaler.transform(test_data.loc[:, feature_vars])

    target_train = target_scaler.transform(train_data.loc[:, target_vars])
    target_val = target_scaler.transform(val_data.loc[:, target_vars])
    target_test = target_scaler.transform(test_data.loc[:, target_vars])

    features = {
        'names': feature_vars,
        'original': np.atleast_2d(train_data.loc[:, feature_vars].to_numpy()),
        'train': np.atleast_2d(feature_train),
        'validation': np.atleast_2d(feature_val),
        'test': np.atleast_2d(feature_test),
        'scaler': feature_scaler,
    }
    targets = {
        'names': target_vars,
        'original': np.atleast_2d(train_data.loc[:, target_vars].to_numpy()),
        'train': np.atleast_2d(target_train),
        'validation': np.atleast_2d(target_val),
        'test': np.atleast_2d(target_test),
        'scaler': target_scaler,
    }

    return features, targets


@tf.function
def ncp_train_epoch(
    model,
    optimizer,
    dataloader,
    loss_function,
    ood_sigmas,
    ood_seed=None,
    verbosity=0
):

    step_total_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'total_loss_array')
    step_likelihood_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'nll_loss_array')
    step_epistemic_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'epi_loss_array')
    step_aleatoric_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'alea_loss_array')

    # Training loop through minibatches - each loop pass is one step
    for nn, (feature_batch, target_batch, epistemic_sigma_batch, aleatoric_sigma_batch) in enumerate(dataloader):

        # Set up training targets into a single large tensor
        target_values = tf.stack([target_batch, tf.zeros(tf.shape(target_batch))], axis=1)
        epistemic_prior_moments = tf.stack([target_batch, epistemic_sigma_batch], axis=1)
        aleatoric_prior_moments = tf.stack([target_batch, aleatoric_sigma_batch], axis=1)
        batch_loss_targets = tf.stack([target_values, epistemic_prior_moments, aleatoric_prior_moments], axis=2)
        n_outputs = batch_loss_targets.shape[-1]

        # Generate random OOD data from training data
        ood_batch_vectors = []
        for jj in range(feature_batch.shape[1]):
            val = feature_batch[:, jj]
            ood = val + tf.random.normal(tf.shape(val), stddev=ood_sigmas[jj], dtype=tf.dtypes.float32, seed=ood_seed)
            ood_batch_vectors.append(ood)
        ood_feature_batch = tf.stack(ood_batch_vectors, axis=-1, name='ood_batch_stack')

        with tf.GradientTape() as tape:

            # For mean data inputs, e.g. training data
            mean_outputs = model(feature_batch, training=True)
            mean_epistemic_avgs = mean_outputs[:, 0, :]
            mean_epistemic_stds = mean_outputs[:, 1, :]
            mean_aleatoric_rngs = mean_outputs[:, 2, :]
            mean_aleatoric_stds = mean_outputs[:, 3, :]

            # For OOD data inputs
            ood_outputs = model(ood_feature_batch, training=True)
            ood_epistemic_avgs = ood_outputs[:, 0, :]
            ood_epistemic_stds = ood_outputs[:, 1, :]
            ood_aleatoric_rngs = ood_outputs[:, 2, :]
            ood_aleatoric_stds = ood_outputs[:, 3, :]

            if tf.executing_eagerly() and verbosity >= 4:
                for ii in range(len(mean_epistemic_avgs)):
                    logger.debug(f'     In-dist model: {mean_epistemic_avgs[ii, 0]}, {mean_epistemic_stds[ii, 0]}')
                    logger.debug(f'     In-dist noise: {mean_aleatoric_rngs[ii, 0]}, {mean_aleatoric_stds[ii, 0]}')
                    logger.debug(f'     Out-of-dist model: {ood_epistemic_avgs[ii, 0]}, {ood_epistemic_stds[ii, 0]}')
                    logger.debug(f'     Out-of-dist noise: {ood_aleatoric_rngs[ii, 0]}, {ood_aleatoric_stds[ii, 0]}')

            # Set up network predictions into equal shape tensor as training targets
            prediction_distributions = tf.stack([mean_aleatoric_rngs, mean_aleatoric_stds], axis=1)
            epistemic_posterior_moments = tf.stack([ood_epistemic_avgs, ood_epistemic_stds], axis=1)
            aleatoric_posterior_moments = tf.stack([ood_aleatoric_rngs, ood_aleatoric_stds], axis=1)
            batch_loss_predictions = tf.stack([prediction_distributions, epistemic_posterior_moments, aleatoric_posterior_moments], axis=2)

            # Compute total loss to be used in adjusting weights and biases
            if n_outputs == 1:
                batch_loss_targets = tf.squeeze(batch_loss_targets, axis=-1)
                batch_loss_predictions = tf.squeeze(batch_loss_predictions, axis=-1)
            step_total_loss = loss_function(batch_loss_targets, batch_loss_predictions)

            # Remaining loss terms purely for inspection purposes
            step_likelihood_loss = loss_function._calculate_likelihood_loss(
                tf.squeeze(tf.gather(batch_loss_targets, indices=[0], axis=2), axis=2),
                tf.squeeze(tf.gather(batch_loss_predictions, indices=[0], axis=2), axis=2)
            )
            step_epistemic_loss = loss_function._calculate_model_divergence_loss(
                tf.squeeze(tf.gather(batch_loss_targets, indices=[1], axis=2), axis=2),
                tf.squeeze(tf.gather(batch_loss_predictions, indices=[1], axis=2), axis=2)
            )
            step_aleatoric_loss = loss_function._calculate_noise_divergence_loss(
                tf.squeeze(tf.gather(batch_loss_targets, indices=[2], axis=2), axis=2),
                tf.squeeze(tf.gather(batch_loss_predictions, indices=[2], axis=2), axis=2)
            )

        # Apply back-propagation
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(step_total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Accumulate batch losses to determine epoch loss
        fill_index = tf.cast(nn + 1, tf.int32)
        step_total_losses = step_total_losses.write(fill_index, tf.reshape(step_total_loss, shape=[-1, 1]))
        step_likelihood_losses = step_likelihood_losses.write(fill_index, tf.reshape(step_likelihood_loss, shape=[-1, n_outputs]))
        step_epistemic_losses = step_epistemic_losses.write(fill_index, tf.reshape(step_epistemic_loss, shape=[-1, n_outputs]))
        step_aleatoric_losses = step_aleatoric_losses.write(fill_index, tf.reshape(step_aleatoric_loss, shape=[-1, n_outputs]))

        if tf.executing_eagerly() and verbosity >= 3:
            logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss:.3f}')
            for ii in range(n_outputs):
                logger.debug(f'     Output {ii}: nll = {step_likelihood_loss[ii]:.3f}, epi = {step_epistemic_loss[ii]:.3f}, alea = {step_aleatoric_loss[ii]:.3f}')

    epoch_total_loss = tf.reduce_sum(step_total_losses.concat(), axis=0)
    epoch_likelihood_loss = tf.reduce_sum(step_likelihood_losses.concat(), axis=0)
    epoch_epistemic_loss = tf.reduce_sum(step_epistemic_losses.concat(), axis=0)
    epoch_aleatoric_loss = tf.reduce_sum(step_aleatoric_losses.concat(), axis=0)

    return epoch_total_loss, epoch_likelihood_loss, epoch_epistemic_loss, epoch_aleatoric_loss


def train_ncp(
    model,
    optimizer,
    features_train,
    targets_train,
    features_valid,
    targets_valid,
    loss_function,
    max_epochs,
    ood_width,
    epi_priors,
    alea_priors,
    batch_size=None,
    patience=None,
    seed=None,
    verbosity=0
):

    n_inputs = features_train.shape[1]
    n_outputs = targets_train.shape[1]
    train_length = features_train.shape[0]
    valid_length = features_valid.shape[0]

    if verbosity >= 1:
        logger.info(f' Number of inputs: {n_inputs}')
        logger.info(f' Number of outputs: {n_outputs}')
        logger.info(f' Training set size: {train_length}')
        logger.info(f' Validation set size: {valid_length}')

    # Assume standardized OOD distribution width based on entire feature value range - better to use quantiles?
    ood_sigma = [ood_width] * n_inputs
    for jj in range(n_inputs):
        ood_sigma[jj] = ood_sigma[jj] * float(np.nanmax(features_train[:, jj]) - np.nanmin(features_train[:, jj]))

    # Create data loaders, including minibatching for training set
    train_data = (features_train.astype(np.float32), targets_train.astype(np.float32), epi_priors.astype(np.float32), alea_priors.astype(np.float32))
    valid_data = (features_valid.astype(np.float32), targets_valid.astype(np.float32))
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data)

    # Create tracker objects to facilitate external analysis of training
    total_tracker = tf.keras.metrics.Sum(name=f'total')
    nll_trackers = []
    epistemic_trackers = []
    aleatoric_trackers = []
    mae_trackers = []
    mse_trackers = []
    for ii in range(n_outputs):
        nll_trackers.append(tf.keras.metrics.Sum(name=f'likelihood{ii}'))
        epistemic_trackers.append(tf.keras.metrics.Sum(name=f'epistemic{ii}'))
        aleatoric_trackers.append(tf.keras.metrics.Sum(name=f'aleatoric{ii}'))
        mae_trackers.append(tf.keras.metrics.MeanAbsoluteError(name=f'mae{ii}'))
        mse_trackers.append(tf.keras.metrics.MeanSquaredError(name=f'mse{ii}'))

    total_list = []
    nll_list = []
    epi_list = []
    alea_list = []
    mae_list = []
    mse_list = []

    # Training loop
    for epoch in range(max_epochs):

        # Training routine described in here
        epoch_total, epoch_nll, epoch_epi, epoch_alea = ncp_train_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            ood_sigma,
            ood_seed=None,
            verbosity=verbosity
        )

        train_outputs = model(train_data[0], training=False)
        train_epistemic_avgs = train_outputs[:, 0, :]
        train_epistemic_stds = train_outputs[:, 1, :]
        train_aleatoric_rngs = train_outputs[:, 2, :]
        train_aleatoric_stds = train_outputs[:, 3, :]

        total_tracker.update_state(epoch_total)
        for ii in range(n_outputs):
            metric_targets = train_data[1][:, ii]
            metric_results = train_epistemic_avgs[:, ii].numpy()
            nll_trackers[ii].update_state(epoch_nll[ii])
            epistemic_trackers[ii].update_state(epoch_epi[ii])
            aleatoric_trackers[ii].update_state(epoch_alea[ii])
            mae_trackers[ii].update_state(metric_targets, metric_results)
            mse_trackers[ii].update_state(metric_targets, metric_results)
        
        if isinstance(patience, int):

            valid_outputs = model(valid_data[0], training=False)
            valid_epistemic_avgs = valid_outputs[:, 0, :]
            valid_epistemic_stds = valid_outputs[:, 1, :]
            valid_aleatoric_rngs = valid_outputs[:, 2, :]
            valid_aleatoric_stds = valid_outputs[:, 3, :]

        total = total_tracker.result().numpy().tolist()
        nll = [np.nan] * n_outputs
        epi = [np.nan] * n_outputs
        alea = [np.nan] * n_outputs
        mae = [np.nan] * n_outputs
        mse = [np.nan] * n_outputs
        for ii in range(n_outputs):
            nll[ii] = nll_trackers[ii].result().numpy().tolist()
            epi[ii] = epistemic_trackers[ii].result().numpy().tolist()
            alea[ii] = aleatoric_trackers[ii].result().numpy().tolist()
            mae[ii] = mae_trackers[ii].result().numpy().tolist()
            mse[ii] = mse_trackers[ii].result().numpy().tolist()

        total_list.append(total)
        nll_list.append(nll)
        epi_list.append(epi)
        alea_list.append(alea)
        mae_list.append(mae)
        mse_list.append(mse)

        print_per_epochs = 100
        if verbosity >= 2:
            print_per_epochs = 10
        if verbosity >= 3:
            print_per_epochs = 1
        if (epoch + 1) % print_per_epochs == 0:
            logger.info(f' Epoch {epoch + 1}: total = {total_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.debug(f'  Output {ii}: mse = {mse_list[-1][ii]:.3f}, mae = {mae_list[-1][ii]:.3f}, nll = {nll_list[-1][ii]:.3f}, epi = {epi_list[-1][ii]:.3f}, alea = {alea_list[-1][ii]:.3f}')

        total_tracker.reset_states()
        for ii in range(n_outputs):
            nll_trackers[ii].reset_states()
            epistemic_trackers[ii].reset_states()
            aleatoric_trackers[ii].reset_states()
            mae_trackers[ii].reset_states()
            mse_trackers[ii].reset_states()

    return total_list, mse_list, mae_list, nll_list, epi_list, alea_list


def train_tensorflow_ncp(
    data_file,
    metrics_file,
    network_file,
    input_vars,
    output_vars,
    validation_fraction=0.1,
    test_fraction=0.1,
    max_epoch=10000,
    batch_size=None,
    early_stopping=None,
    shuffle_seed=None,
    sample_seed=None,
    generalized_widths=None,
    specialized_depths=None,
    specialized_widths=None,
    ood_sampling_width=0.2,
    epistemic_priors=None,
    aleatoric_priors=None,
    likelihood_weights=None,
    epistemic_weights=None,
    aleatoric_weights=None,
    learning_rate=0.001,
    decay_epoch=0.98,
    decay_rate=20,
    disable_gpu=False,
    log_file=None,
    verbosity=0
):

    settings = {
        'data_file': data_file,
        'metrics_file': metrics_file,
        'network_file': network_file,
        'input_vars': input_vars,
        'output_vars': output_vars,
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'shuffle_seed': shuffle_seed,
        'sample_seed': sample_seed,
        'generalized_widths': generalized_widths,
        'specialized_depths': specialized_depths,
        'specialized_widths': specialized_widths,
        'ood_sampling_width': ood_sampling_width,
        'epistemic_priors': epistemic_priors,
        'aleatoric_priors': aleatoric_priors,
        'likelihood_weights': likelihood_weights,
        'epistemic_weights': epistemic_weights,
        'aleatoric_weights': aleatoric_weights,
        'learning_rate': learning_rate,
        'decay_epoch': decay_epoch,
        'decay_rate': decay_rate,
        'disable_gpu': disable_gpu,
        'log_file': log_file,
        'verbosity': verbosity,
    }

    lpath = Path(log_file) if isinstance(log_file, str) else None
    setup_logging(lpath, verbosity)
    if verbosity >= 2:
        print_settings(settings)

    ipath = Path(data_file)
    mpath = Path(metrics_file)
    npath = Path(network_file)
    if ipath.is_file():

        if disable_gpu:
            tf.config.set_visible_devices([], 'GPU')

        if verbosity >= 2:
            tf.config.run_functions_eagerly(True)

        # Set up the required data sets
        start_preprocess = time.perf_counter()
        data = pd.read_hdf(ipath, key='/data')
        features, targets = preprocess_data(
            data,
            input_vars,
            output_vars,
            validation_fraction,
            test_fraction,
            seed=shuffle_seed,
            verbosity=verbosity
        )
        end_preprocess = time.perf_counter()

        logger.info(f'Pre-processing completed! Elpased time: {(end_preprocess - start_preprocess):.4f} s')

        # Set up the BNN-NCP model
        start_setup = time.perf_counter()
        n_inputs = features['train'].shape[1]
        n_outputs = targets['train'].shape[1]
        n_commons = len(generalized_widths) if isinstance(generalized_widths, (list, tuple)) else 0
        common_nodes = list(generalized_widths) if n_commons > 0 else None
        special_nodes = None
        if isinstance(specialized_depths, (list, tuple)) and len(specialized_depths) > 0:
            special_nodes = []
            kk = 0
            for jj in range(len(specialized_depths)):
                output_special_nodes = []
                if isinstance(specialized_widths, (list, tuple)) and kk < len(specialized_widths):
                    output_special_nodes.append(specialized_widths[kk])
                    kk += 1
                special_nodes.append(output_special_nodes)   # List of lists
        model = create_model(
            n_input=n_inputs,
            n_output=n_outputs,
            n_common=n_commons,
            common_nodes=common_nodes,
            special_nodes=special_nodes,
            style='ncp',
            verbosity=verbosity
        )

        # Set up the user-defined prior factors, default behaviour included if input is None
        epi_priors = 0.001 * targets['original'] / targets['scaler'].scale_
        for ii in range(n_outputs):
            epi_factor = 0.001
            if isinstance(epistemic_priors, list):
                epi_factor = epistemic_priors[ii] if ii < len(epistemic_priors) else epistemic_priors[-1]
            epi_priors[:, ii] = np.abs(epi_factor * targets['original'][:, ii] / targets['scaler'].scale_[ii])
        alea_priors = 0.001 * targets['original'] / targets['scaler'].scale_
        for ii in range(n_outputs):
            alea_factor = 0.001
            if isinstance(aleatoric_priors, list):
                alea_factor = aleatoric_priors[ii] if ii < len(aleatoric_priors) else aleatoric_priors[-1]
            alea_priors[:, ii] = np.abs(alea_factor * targets['original'][:, ii] / targets['scaler'].scale_[ii])

        # Required minimum priors to avoid infs and nans in KL-divergence
        epi_priors[epi_priors < 1.0e-6] = 1.0e-6
        alea_priors[alea_priors < 1.0e-6] = 1.0e-6

        # Set up the user-defined loss term weights, default behaviour included if input is None
        nll_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(likelihood_weights, list):
                nll_weights[ii] = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
        epi_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(epistemic_weights, list):
                epi_weights[ii] = epistemic_weights[ii] if ii < len(epistemic_weights) else epistemic_weights[-1]
        alea_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(aleatoric_weights, list):
                alea_weights[ii] = aleatoric_weights[ii] if ii < len(aleatoric_weights) else aleatoric_weights[-1]

        # Create custom loss function, weights converted into tensor objects internally
        loss_function = create_loss_function(
            n_outputs,
            style='ncp',
            nll_weights=nll_weights,
            epi_weights=epi_weights,
            alea_weights=alea_weights,
            verbosity=verbosity
        )

        train_length = features['train'].shape[0]
        steps_per_epoch = int(np.ceil(train_length / batch_size)) if isinstance(batch_size, int) else 1
        decay_steps = steps_per_epoch * decay_epoch
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
        total_list, mse_list, mae_list, nll_list, epistemic_list, aleatoric_list = train_ncp(
            model,
            optimizer,
            features['train'],
            targets['train'],
            features['validation'],
            targets['validation'],
            loss_function,
            max_epoch,
            ood_sampling_width,
            epi_priors,
            alea_priors,
            batch_size=batch_size,
            patience=early_stopping,
            seed=sample_seed,
            verbosity=verbosity
        )
        end_train = time.perf_counter()

        logger.info(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

        # Save the trained model and training metrics
        start_save = time.perf_counter()
        total = np.array(total_list)
        mse = np.atleast_2d(mse_list)
        mae = np.atleast_2d(mae_list)
        nll = np.atleast_2d(nll_list)
        epistemic = np.atleast_2d(epistemic_list)
        aleatoric = np.atleast_2d(aleatoric_list)
        metric_dict = {'total': total.flatten()}
        for ii in range(n_outputs):
            metric_dict[f'mse{ii}'] = mse[:, ii].flatten()
            metric_dict[f'mae{ii}'] = mae[:, ii].flatten()
            metric_dict[f'nll{ii}'] = nll[:, ii].flatten()
            metric_dict[f'epi{ii}'] = epistemic[:, ii].flatten()
            metric_dict[f'alea{ii}'] = aleatoric[:, ii].flatten()
        metrics = pd.DataFrame(data=metric_dict)
        metrics.to_hdf(mpath, key='/data')
        descaled_model = wrap_model(model, features['scaler'], targets['scaler'])
        descaled_model.save(npath)
        end_save = time.perf_counter()

        logger.info(f'Saving completed! Elapsed time: {(end_save - start_save):.4f} s')

        logger.info(f' Metrics saved in {mpath}')
        logger.info(f' Network saved in {npath}')

        logger.info(f'Script completed!')

    else:
        raise IOError(f'Could not find input file: {ipath}')


def main():

    args = parse_inputs()
    train_tensorflow_ncp(
        data_file=args.data_file,
        metrics_file=args.metrics_file,
        network_file=args.network_file,
        input_vars=args.input_var,
        output_vars=args.output_var,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        shuffle_seed=args.shuffle_seed,
        sample_seed=args.sample_seed,
        generalized_widths=args.generalized_node,
        specialized_depths=args.specialized_layer,
        specialized_widths=args.specialized_node,
        ood_sampling_width=args.ood_width,
        epistemic_priors=args.epi_prior,
        aleatoric_priors=args.alea_prior,
        likelihood_weights=args.nll_weight,
        epistemic_weights=args.epi_weight,
        aleatoric_weights=args.alea_weight,
        learning_rate=args.learning_rate,
        decay_epoch=args.decay_epoch,
        decay_rate=args.decay_rate,
        disable_gpu=args.disable_gpu,
        log_file=args.log_file,
        verbosity=args.verbosity
    )


if __name__ == "__main__":
    main()

