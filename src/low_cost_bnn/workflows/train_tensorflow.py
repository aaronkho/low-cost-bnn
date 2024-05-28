import sys
import argparse
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from ..utils.helpers import create_scaler, split
from ..utils.helpers_tensorflow import create_data_loader, create_scheduled_adam_optimizer, create_model, create_loss_function, wrap_model

logger = logging.getLogger("train_tensorflow")

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Input HDF5 file containing training data set')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of HDF5 file to store training metrics')
    parser.add_argument('--input_vars', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_vars', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--sample_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for OOD sampling')
    parser.add_argument('--hidden_nodes', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the common hidden layer')
    parser.add_argument('--specialized_nodes', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layer')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--max_epochs', metavar='n', type=int, default=10000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=None, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--ood_width', metavar='val', type=float, default=0.2, help='Normalized standard deviation of OOD sampling distribution')
    parser.add_argument('--epi_prior', metavar='val', type=float, nargs='*', default=None, help='Standard deviation of epistemic priors used to compute epistemic loss term')
    parser.add_argument('--alea_prior', metavar='val', type=float, nargs='*', default=None, help='Standard deviation of aleatoric priors used to compute aleatoric loss term')
    parser.add_argument('--nll_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the NLL loss term')
    parser.add_argument('--epi_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to epistemic loss term')
    parser.add_argument('--alea_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to aleatoric loss term')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.98, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epochs', metavar='n', type=float, default=20, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to log file where script related print outs will be stored')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def print_settings(args):
    logger.debug(f'Input settings print-out requested...')
    for key, val in vars(args).items():
       logger.debug(f'  {key}: {val}')


def setup_logging(log_file=None, verbosity=0):

    formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    if isinstance(log_file, str):
        log = logging.FileHandler(log_file, mode='w')
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
        'original': train_data.loc[:, feature_vars].to_numpy(),
        'train': feature_train,
        'validation': feature_val,
        'test': feature_test,
        'scaler': feature_scaler,
    }
    targets = {
        'names': target_vars,
        'original': train_data.loc[:, target_vars].to_numpy(),
        'train': target_train,
        'validation': target_val,
        'test': target_test,
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

        # Define epistemic prior for NCP methodology
        epistemic_priors = []
        for ii in range(target_batch.shape[1]):
            target_reshape = tf.expand_dims(target_batch[:, ii], axis=-1)
            sigma_reshape = tf.expand_dims(epistemic_sigma_batch[:, ii], axis=-1)
            prior = tfd.Normal(target_reshape, sigma_reshape, name=f'EpiPrior{ii}')
            epistemic_priors.append(prior)

        # Define aleatoric prior for NCP methodology
        aleatoric_priors = []
        for ii in range(target_batch.shape[1]):
            target_reshape = tf.expand_dims(target_batch[:, ii], axis=-1)
            sigma_reshape = tf.expand_dims(aleatoric_sigma_batch[:, ii], axis=-1)
            prior = tfd.Normal(target_reshape, sigma_reshape, name=f'AleaPrior{ii}')
            aleatoric_priors.append(prior)
        
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
            n_outputs = mean_outputs.shape[1] / 4
            mean_epistemic_avgs = tf.reshape(mean_outputs[:, 0::4], shape=tf.shape((None, n_outputs)))
            mean_epistemic_stds = tf.reshape(mean_outputs[:, 1::4], shape=tf.shape((None, n_outputs)))
            mean_aleatoric_rngs = tf.reshape(mean_outputs[:, 2::4], shape=tf.shape((None, n_outputs)))
            mean_aleatoric_stds = tf.reshape(mean_outputs[:, 3::4], shape=tf.shape((None, n_outputs)))

            # For OOD data inputs
            ood_outputs = model(ood_feature_batch, training=True)
            ood_epistemic_avgs = tf.reshape(ood_outputs[:, 0::4], shape=tf.shape((None, n_outputs)))
            ood_epistemic_stds = tf.reshape(ood_outputs[:, 1::4], shape=tf.shape((None, n_outputs)))
            ood_aleatoric_rngs = tf.reshape(ood_outputs[:, 2::4], shape=tf.shape((None, n_outputs)))
            ood_aleatoric_stds = tf.reshape(ood_outputs[:, 3::4], shape=tf.shape((None, n_outputs)))

            if tf.executing_eagerly() and verbosity >= 4:
                for ii in range(len(mean_epistemic_avgs)):
                    logger.debug(f'     In-dist model: {mean_epistemic_avgs[ii, 0]}, {mean_epistemic_stds[ii, 0]}')
                    logger.debug(f'     In-dist noise: {mean_aleatoric_rngs[ii, 0]}, {mean_aleatoric_stds[ii, 0]}')
                    logger.debug(f'     Out-of-dist model: {ood_epistemic_avgs[ii, 0]}, {ood_epistemic_stds[ii, 0]}')
                    logger.debug(f'     Out-of-dist noise: {ood_aleatoric_rngs[ii, 0]}, {ood_aleatoric_stds[ii, 0]}')

            mean_epistemic_dists = []
            mean_aleatoric_dists = []
            for ii in range(mean_epistemic_avgs.shape[1]):
                mean_epistemic_dists.append(tfd.Normal(mean_epistemic_avgs[:, ii], mean_epistemic_stds[:, ii], name=f'EpiPosterior{ii}'))
                mean_aleatoric_dists.append(tfd.Normal(mean_aleatoric_rngs[:, ii], mean_aleatoric_stds[:, ii], name=f'AleaPosterior{ii}'))

            ood_epistemic_dists = []
            ood_aleatoric_dists = []
            for ii in range(ood_epistemic_avgs.shape[1]):
                ood_epistemic_dists.append(tfd.Normal(ood_epistemic_avgs[:, ii], ood_epistemic_stds[:, ii], name=f'EpiOODPosterior{ii}'))
                ood_aleatoric_dists.append(tfd.Normal(ood_aleatoric_rngs[:, ii], ood_aleatoric_stds[:, ii], name=f'AleaOODPosterior{ii}'))

            # Compute total loss to be used in adjusting weights and biases
            target_lists = [target_batch[:, jj] for jj in range(target_batch.shape[1])]
            step_total_loss = loss_function.call(target_lists, mean_aleatoric_dists, epistemic_priors, ood_epistemic_dists, aleatoric_priors, ood_aleatoric_dists)

            # Remaining loss terms purely for inspection purposes
            step_likelihood_loss = loss_function._calculate_likelihood_loss(target_lists, mean_aleatoric_dists)
            step_epistemic_loss = loss_function._calculate_model_divergence_loss(epistemic_priors, ood_epistemic_dists)
            step_aleatoric_loss = loss_function._calculate_noise_divergence_loss(aleatoric_priors, ood_aleatoric_dists)

        # Apply back-propagation
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(step_total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Accumulate batch losses to determine epoch loss
        fill_index = tf.cast(nn + 1, tf.int32)
        step_total_losses = step_total_losses.write(fill_index, step_total_loss)
        step_likelihood_losses = step_likelihood_losses.write(fill_index, step_likelihood_loss)
        step_epistemic_losses = step_epistemic_losses.write(fill_index, step_epistemic_loss)
        step_aleatoric_losses = step_aleatoric_losses.write(fill_index, step_aleatoric_loss)

        if tf.executing_eagerly() and verbosity >= 3:
            logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss:.3f}')
            for ii in range(len(mean_epistemic_dists)):
                logger.debug(f'     Output {ii}: nll = {step_likelihood_loss[ii]:.3f}, epi = {step_epistemic_loss[ii]:.3f}, alea = {step_aleatoric_loss[ii]:.3f}')

    epoch_total_loss = tf.reduce_sum(step_total_losses.stack(), axis=0)
    epoch_likelihood_loss = tf.reduce_sum(step_likelihood_losses.stack(), axis=0)
    epoch_epistemic_loss = tf.reduce_sum(step_epistemic_losses.stack(), axis=0)
    epoch_aleatoric_loss = tf.reduce_sum(step_aleatoric_losses.stack(), axis=0)

    return epoch_total_loss, epoch_likelihood_loss, epoch_epistemic_loss, epoch_aleatoric_loss


def train(
    model,
    optimizer,
    features_train,
    targets_train,
    features_valid,
    targets_valid,
    max_epochs,
    ood_width,
    epi_priors,
    alea_priors,
    nll_weights,
    epi_weights,
    alea_weights,
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

    # Convert loss term weights into tensor objects
    #nll_weight_tensors = []
    #for ii in range(len(nll_weights)):
    #    factor = tf.constant(nll_weights[ii], dtype=tf.dtypes.float32, name=f'NLLWeight{ii}')
    #    nll_weight_tensors.append(factor)
    #epi_weight_tensors = []
    #for ii in range(len(epi_weights)):
    #    factor = tf.constant(epi_weights[ii], dtype=tf.dtypes.float32, name=f'EpiWeight{ii}')
    #    epi_weight_tensors.append(factor)
    #alea_weight_tensors = []
    #for ii in range(len(alea_weights)):
    #    factor = tf.constant(alea_weights[ii], dtype=tf.dtypes.float32, name='AleaWeight{ii}')
    #    alea_weight_tensors.append(factor)

    # Create custom loss function, weights converted into tensor objects internally
    loss_function = create_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, verbosity=verbosity)

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
        total, nll, epi, alea = ncp_train_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            #nll_weight_tensors,
            #epi_weight_tensors,
            #alea_weight_tensors,
            ood_sigma,
            ood_seed=None,
            verbosity=verbosity
        )

        train_outputs = model(train_data[0], training=False)
        train_model_dists = train_outputs[::2]
        train_noise_dists = train_outputs[1::2]

        total_tracker.update_state(total)
        for ii in range(n_outputs):
            metric_targets = np.atleast_2d(train_data[1][:, ii]).T
            metric_results = train_model_dists[ii].mean()
            nll_trackers[ii].update_state(nll[ii])
            epistemic_trackers[ii].update_state(epi[ii])
            aleatoric_trackers[ii].update_state(alea[ii])
            mae_trackers[ii].update_state(metric_targets, metric_results)
            mse_trackers[ii].update_state(metric_targets, metric_results)
        
        if isinstance(patience, int):

            valid_outputs = model(valid_data[0], training=False)
            valid_model_dists = valid_outputs[::2]
            valid_noise_dists = valid_outputs[1::2]

        nll = [np.nan] * n_outputs
        epi = [np.nan] * n_outputs
        alea = [np.nan] * n_outputs
        mae = [np.nan] * n_outputs
        mse = [np.nan] * n_outputs
        for ii in range(n_outputs):
            nll[ii] = nll_trackers[ii].result().numpy()
            epi[ii] = epistemic_trackers[ii].result().numpy()
            alea[ii] = aleatoric_trackers[ii].result().numpy()
            mae[ii] = mae_trackers[ii].result().numpy()
            mse[ii] = mse_trackers[ii].result().numpy()

        total_list.append(total_tracker.result().numpy())
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


def main():

    args = parse_inputs()
    setup_logging(args.log_file, args.verbosity)
    if args.verbosity >= 2:
        print_settings(args)

    ipath = Path(args.data_file)
    mpath = Path(args.metrics_file)
    npath = Path(args.network_file)
    if ipath.is_file():

        if args.verbosity >= 2:
            tf.config.run_functions_eagerly(True)

        # Set up the required data sets
        start_preprocess = time.perf_counter()
        data = pd.read_hdf(ipath, '/data')
        features, targets = preprocess_data(
            data,
            args.input_vars,
            args.output_vars,
            args.validation_fraction,
            args.test_fraction,
            seed=args.shuffle_seed,
            verbosity=args.verbosity
        )
        end_preprocess = time.perf_counter()

        logger.info(f'Pre-processing completed! Elpased time: {(end_preprocess - start_preprocess):.4f} s')

        # Set up the BNN-NCP model
        start_setup = time.perf_counter()
        n_inputs = features['train'].shape[1]
        n_outputs = targets['train'].shape[1]
        model = create_model(
            n_input=n_inputs,
            n_output=n_outputs,
            n_hidden=args.hidden_nodes,
            n_special=args.specialized_nodes,
            verbosity=args.verbosity
        )

        # Set up the user-defined prior factors, default behaviour included if input is None
        epi_priors = 0.001 * features['original'] / features['scaler'].scale_
        for ii in range(n_outputs):
            epi_factor = 0.001
            if isinstance(args.epi_prior, list):
                epi_factor = args.epi_prior[ii] if ii < len(args.epi_prior) else args.epi_prior[-1]
            epi_priors[:, ii] = np.abs(epi_factor * features['original'][:, ii] / features['scaler'].scale_[ii])
        alea_priors = 0.001 * features['original'] / features['scaler'].scale_
        for ii in range(n_outputs):
            alea_factor = 0.001
            if isinstance(args.alea_prior, list):
                alea_factor = args.alea_prior[ii] if ii < len(args.alea_prior) else args.alea_prior[-1]
            alea_priors[:, ii] = np.abs(alea_factor * features['original'][:, ii] / features['scaler'].scale_[ii])

        # Required minimum priors to avoid infs and nans in KL-divergence
        epi_priors[epi_priors < 1.0e-6] = 1.0e-6
        alea_priors[alea_priors < 1.0e-6] = 1.0e-6

        # Set up the user-defined loss term weights, default behaviour included if input is None
        nll_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.nll_weight, list):
                nll_weights[ii] = args.nll_weight[ii] if ii < len(args.nll_weight) else args.nll_weight[-1]
        epi_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.epi_weight, list):
                epi_weights[ii] = args.epi_weight[ii] if ii < len(args.epi_weight) else args.epi_weight[-1]
        alea_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.alea_weight, list):
                alea_weights[ii] = args.alea_weight[ii] if ii < len(args.alea_weight) else args.alea_weight[-1]

        train_length = features['train'].shape[0]
        steps_per_epoch = int(np.ceil(train_length / args.batch_size)) if isinstance(args.batch_size, int) else 1
        steps = steps_per_epoch * args.decay_epochs
        optimizer, scheduler = create_scheduled_adam_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            decay_steps=steps,
            decay_rate=args.decay_rate
        )
        end_setup = time.perf_counter()

        logger.info(f'Setup completed! Elapsed time: {(end_setup - start_setup):.4f} s')

        # Perform the training loop
        start_train = time.perf_counter()
        total_list, mse_list, mae_list, nll_list, epistemic_list, aleatoric_list = train(
            model,
            optimizer,
            features['train'],
            targets['train'],
            features['validation'],
            targets['validation'],
            args.max_epochs,
            args.ood_width,
            epi_priors,
            alea_priors,
            nll_weights,
            epi_weights,
            alea_weights,
            batch_size=args.batch_size,
            patience=args.early_stopping,
            seed=args.sample_seed,
            verbosity=args.verbosity
        )
        end_train = time.perf_counter()

        logger.info(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

        # Save the trained model and training metrics
        start_save = time.perf_counter()
        total = np.atleast_2d(total_list)
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


if __name__ == "__main__":
    main()
