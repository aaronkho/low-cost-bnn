import argparse
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from ..utils.pipeline_tools import setup_logging, print_settings, preprocess_data
from ..utils.helpers_tensorflow import create_data_loader, create_scheduled_adam_optimizer, create_model, create_loss_function, wrap_model

logger = logging.getLogger("train_tensorflow")

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Path and name of input HDF5 file containing training data set')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of output HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of output file to store training metrics')
    parser.add_argument('--input_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--max_epoch', metavar='n', type=int, default=10000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=None, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--generalized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the generalized hidden layers')
    parser.add_argument('--specialized_layer', metavar='n', type=int, nargs='*', default=None, help='Number of specialized hidden layers, given for each output')
    parser.add_argument('--specialized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layers, sequential per output stack')
    parser.add_argument('--nll_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the NLL loss term')
    parser.add_argument('--reg_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to regularization loss term')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.98, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epoch', metavar='n', type=float, default=20, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to log file where script related print outs will be stored')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


@tf.function
def train_tensorflow_evidential_epoch(
    model,
    optimizer,
    dataloader,
    loss_function,
    training=True,
    verbosity=0
):

    step_total_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'total_loss_array')
    step_likelihood_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'nll_loss_array')
    step_regularization_losses = tf.TensorArray(dtype=tf.dtypes.float32, size=0, dynamic_size=True, clear_after_read=True, name=f'epi_loss_array')

    # Training loop through minibatches - each loop pass is one step
    for nn, (feature_batch, target_batch) in enumerate(dataloader):

        # Set up training targets into a single large tensor
        target_values = tf.stack([target_batch, tf.zeros(tf.shape(target_batch)), tf.zeros(tf.shape(target_batch)), tf.zeros(tf.shape(target_batch))], axis=1)
        batch_loss_targets = tf.stack([target_values, target_values], axis=2)
        n_outputs = batch_loss_targets.shape[-1]

        with tf.GradientTape() as tape:

            # For mean data inputs, e.g. training data
            outputs = model(feature_batch, training=training)

            if training and tf.executing_eagerly() and verbosity >= 4:
                for ii in range(n_outputs):
                    logger.debug(f'     gamma: {outputs[0, 0, ii]}')
                    logger.debug(f'     nu: {outputs[0, 1, ii]}')
                    logger.debug(f'     alpha: {outputs[0, 2, ii]}')
                    logger.debug(f'     beta: {outputs[0, 3, ii]}')

            # Set up network predictions into equal shape tensor as training targets
            batch_loss_predictions = tf.stack([outputs, outputs], axis=2)

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
            step_regularization_loss = loss_function._calculate_regularization_loss(
                tf.squeeze(tf.gather(batch_loss_targets, indices=[1], axis=2), axis=2),
                tf.squeeze(tf.gather(batch_loss_predictions, indices=[1], axis=2), axis=2)
            )

        # Apply back-propagation
        if training:
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(step_total_loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Accumulate batch losses to determine epoch loss
        fill_index = tf.cast(nn + 1, tf.int32)
        step_total_losses = step_total_losses.write(fill_index, tf.reshape(step_total_loss, shape=[-1, 1]))
        step_likelihood_losses = step_likelihood_losses.write(fill_index, tf.reshape(step_likelihood_loss, shape=[-1, n_outputs]))
        step_regularization_losses = step_regularization_losses.write(fill_index, tf.reshape(step_regularization_loss, shape=[-1, n_outputs]))

        if tf.executing_eagerly() and verbosity >= 3:
            if training:
                logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss:.3f}')
                for ii in range(n_outputs):
                    logger.debug(f'     Output {ii}: nll = {step_likelihood_loss[ii]:.3f}, reg = {step_regularization_loss[ii]:.3f}')
            else:
                logger.debug(f'  - Validation: total = {step_total_loss:.3f}')
                for ii in range(n_outputs):
                    logger.debug(f'     Output {ii}: nll = {step_likelihood_loss[ii]:.3f}, reg = {step_regularization_loss[ii]:.3f}')

    epoch_total_loss = tf.reduce_sum(step_total_losses.concat(), axis=0)
    epoch_likelihood_loss = tf.reduce_sum(step_likelihood_losses.concat(), axis=0)
    epoch_regularization_loss = tf.reduce_sum(step_regularization_losses.concat(), axis=0)

    return epoch_total_loss, epoch_likelihood_loss, epoch_regularization_loss


def train_tensorflow_evidential(
    model,
    optimizer,
    features_train,
    targets_train,
    features_valid,
    targets_valid,
    loss_function,
    max_epochs,
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

    # Create data loaders, including minibatching for training set
    train_data = (features_train.astype(np.float32), targets_train.astype(np.float32))
    valid_data = (features_valid.astype(np.float32), targets_valid.astype(np.float32))
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data, batch_size=valid_length)

    # Create training tracker objects to facilitate external analysis of pipeline
    total_train_tracker = tf.keras.metrics.Sum(name=f'train_total')
    nll_train_trackers = []
    reg_train_trackers = []
    mae_train_trackers = []
    mse_train_trackers = []
    for ii in range(n_outputs):
        nll_train_trackers.append(tf.keras.metrics.Sum(name=f'train_likelihood{ii}'))
        reg_train_trackers.append(tf.keras.metrics.Sum(name=f'train_regularization{ii}'))
        mae_train_trackers.append(tf.keras.metrics.MeanAbsoluteError(name=f'train_mae{ii}'))
        mse_train_trackers.append(tf.keras.metrics.MeanSquaredError(name=f'train_mse{ii}'))

    # Create validation tracker objects to facilitate external analysis of pipeline
    total_valid_tracker = tf.keras.metrics.Sum(name=f'valid_total')
    nll_valid_trackers = []
    reg_valid_trackers = []
    mae_valid_trackers = []
    mse_valid_trackers = []
    for ii in range(n_outputs):
        nll_valid_trackers.append(tf.keras.metrics.Sum(name=f'valid_likelihood{ii}'))
        reg_valid_trackers.append(tf.keras.metrics.Sum(name=f'valid_regularization{ii}'))
        mae_valid_trackers.append(tf.keras.metrics.MeanAbsoluteError(name=f'valid_mae{ii}'))
        mse_valid_trackers.append(tf.keras.metrics.MeanSquaredError(name=f'valid_mse{ii}'))

    # Output containers
    total_train_list = []
    nll_train_list = []
    reg_train_list = []
    mae_train_list = []
    mse_train_list = []
    total_valid_list = []
    nll_valid_list = []
    reg_valid_list = []
    mae_valid_list = []
    mse_valid_list = []

    # Training loop
    for epoch in range(max_epochs):

        # Training routine described in here
        epoch_total, epoch_nll, epoch_reg = train_tensorflow_evidential_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            training=True,
            verbosity=verbosity
        )

        # Evaluate model with full training data set for performance tracking
        train_outputs = model(train_data[0], training=False)
        train_means = tf.squeeze(tf.gather(train_outputs, indices=[0], axis=1), axis=1)

        total_train_tracker.update_state(epoch_total / train_length)
        for ii in range(n_outputs):
            metric_targets = train_data[1][:, ii]
            metric_results = train_means[:, ii].numpy()
            nll_train_trackers[ii].update_state(epoch_nll[ii] / train_length)
            reg_train_trackers[ii].update_state(epoch_reg[ii] / train_length)
            mae_train_trackers[ii].update_state(metric_targets, metric_results)
            mse_train_trackers[ii].update_state(metric_targets, metric_results)

        total_train = total_train_tracker.result().numpy().tolist()
        nll_train = [np.nan] * n_outputs
        reg_train = [np.nan] * n_outputs
        mae_train = [np.nan] * n_outputs
        mse_train = [np.nan] * n_outputs
        for ii in range(n_outputs):
            nll_train[ii] = nll_train_trackers[ii].result().numpy().tolist()
            reg_train[ii] = reg_train_trackers[ii].result().numpy().tolist()
            mae_train[ii] = mae_train_trackers[ii].result().numpy().tolist()
            mse_train[ii] = mse_train_trackers[ii].result().numpy().tolist()

        total_train_list.append(total_train)
        nll_train_list.append(nll_train)
        reg_train_list.append(reg_train)
        mae_train_list.append(mae_train)
        mse_train_list.append(mse_train)

        # Reuse training routine to evaluate validation data
        valid_total, valid_nll, valid_reg = train_tensorflow_evidential_epoch(
            model,
            optimizer,
            valid_loader,
            loss_function,
            training=False,
            verbosity=verbosity
        )

        # Evaluate model with validation data set for performance tracking
        valid_outputs = model(valid_data[0], training=False)
        valid_means = tf.squeeze(tf.gather(valid_outputs, indices=[0], axis=1), axis=1)

        total_valid_tracker.update_state(valid_total / valid_length)
        for ii in range(n_outputs):
            metric_targets = valid_data[1][:, ii]
            metric_results = valid_means[:, ii].numpy()
            nll_valid_trackers[ii].update_state(valid_nll[ii] / valid_length)
            reg_valid_trackers[ii].update_state(valid_reg[ii] / valid_length)
            mae_valid_trackers[ii].update_state(metric_targets, metric_results)
            mse_valid_trackers[ii].update_state(metric_targets, metric_results)

        total_valid = total_valid_tracker.result().numpy().tolist()
        nll_valid = [np.nan] * n_outputs
        reg_valid = [np.nan] * n_outputs
        mae_valid = [np.nan] * n_outputs
        mse_valid = [np.nan] * n_outputs
        for ii in range(n_outputs):
            nll_valid[ii] = nll_valid_trackers[ii].result().numpy().tolist()
            reg_valid[ii] = reg_valid_trackers[ii].result().numpy().tolist()
            mae_valid[ii] = mae_valid_trackers[ii].result().numpy().tolist()
            mse_valid[ii] = mse_valid_trackers[ii].result().numpy().tolist()

        total_valid_list.append(total_valid)
        nll_valid_list.append(nll_valid)
        reg_valid_list.append(reg_valid)
        mae_valid_list.append(mae_valid)
        mse_valid_list.append(mse_valid)

        if isinstance(patience, int):

            pass

        print_per_epochs = 100
        if verbosity >= 2:
            print_per_epochs = 10
        if verbosity >= 3:
            print_per_epochs = 1
        if (epoch + 1) % print_per_epochs == 0:
            logger.info(f' Epoch {epoch + 1}: total_train = {total_train_list[-1]:.3f}, valid_total = {total_valid_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.debug(f'  Train: Output {ii}: mse = {mse_train_list[-1][ii]:.3f}, mae = {mae_train_list[-1][ii]:.3f}, nll = {nll_train_list[-1][ii]:.3f}, reg = {reg_train_list[-1][ii]:.3f}')
                logger.debug(f'  Valid: Output {ii}: mse = {mse_valid_list[-1][ii]:.3f}, mae = {mae_valid_list[-1][ii]:.3f}, nll = {nll_valid_list[-1][ii]:.3f}, reg = {reg_valid_list[-1][ii]:.3f}')

        total_train_tracker.reset_states()
        total_valid_tracker.reset_states()
        for ii in range(n_outputs):
            nll_train_trackers[ii].reset_states()
            reg_train_trackers[ii].reset_states()
            mae_train_trackers[ii].reset_states()
            mse_train_trackers[ii].reset_states()
            nll_valid_trackers[ii].reset_states()
            reg_valid_trackers[ii].reset_states()
            mae_valid_trackers[ii].reset_states()
            mse_valid_trackers[ii].reset_states()

    metrics_dict = {
        'train_total': total_train_list,
        'valid_total': total_valid_list,
        'train_mse': mse_train_list,
        'train_mae': mae_train_list,
        'train_nll': nll_train_list,
        'train_reg': reg_train_list,
        'valid_mse': mse_valid_list,
        'valid_mae': mae_valid_list,
        'valid_nll': nll_valid_list,
        'valid_reg': reg_valid_list
    }

    return metrics_dict


def launch_tensorflow_pipeline_evidential(
    data,
    input_vars,
    output_vars,
    validation_fraction=0.1,
    test_fraction=0.1,
    max_epoch=10000,
    batch_size=None,
    early_stopping=None,
    shuffle_seed=None,
    generalized_widths=None,
    specialized_depths=None,
    specialized_widths=None,
    likelihood_weights=None,
    regularization_weights=None,
    learning_rate=0.001,
    decay_epoch=0.98,
    decay_rate=20,
    verbosity=0
):

    settings = {
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'shuffle_seed': shuffle_seed,
        'generalized_widths': generalized_widths,
        'specialized_depths': specialized_depths,
        'specialized_widths': specialized_widths,
        'likelihood_weights': likelihood_weights,
        'regularization_weights': regularization_weights,
        'learning_rate': learning_rate,
        'decay_epoch': decay_epoch,
        'decay_rate': decay_rate,
    }

    if verbosity >= 2:
        print_settings(logger, settings, 'Evidential model and training settings:')

    # Set up the required data sets
    start_preprocess = time.perf_counter()
    features, targets = preprocess_data(
        data,
        input_vars,
        output_vars,
        validation_fraction,
        test_fraction,
        seed=shuffle_seed,
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
        style='evidential',
        verbosity=verbosity
    )

    # Set up the user-defined loss term weights, default behaviour included if input is None
    nll_weights = [1.0] * n_outputs
    for ii in range(n_outputs):
        if isinstance(likelihood_weights, list):
            nll_weights[ii] = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
    reg_weights = [1.0] * n_outputs
    for ii in range(n_outputs):
        if isinstance(regularization_weights, list):
            reg_weights[ii] = regularization_weights[ii] if ii < len(regularization_weights) else regularization_weights[-1]

    # Create custom loss function, weights converted into tensor objects internally
    loss_function = create_loss_function(
        n_outputs,
        style='evidential',
        nll_weights=nll_weights,
        reg_weights=reg_weights,
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
    metrics = train_tensorflow_evidential(
        model,
        optimizer,
        features['train'],
        targets['train'],
        features['validation'],
        targets['validation'],
        loss_function,
        max_epoch,
        batch_size=batch_size,
        patience=early_stopping,
        verbosity=verbosity
    )
    end_train = time.perf_counter()

    logger.info(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

    # Configure the trained model and training metrics for saving
    start_out = time.perf_counter()
    metrics_dict = {}
    for key, val in metrics.items():
        if key.endswith('total'):
            metric = np.array(val)
            metrics_dict[f'{key}'] = metric.flatten()
        else:
            metric = np.atleast_2d(val)
            for ii in range(n_outputs):
                metrics_dict[f'{key}{ii}'] = metric[:, ii].flatten()
    metrics_df = pd.DataFrame(data=metrics_dict)
    wrapped_model = wrap_model(model, features['scaler'], targets['scaler'])
    end_out = time.perf_counter()

    logger.info(f'Output configuration completed! Elapsed time: {(end_out - start_out):.4f} s')

    return wrapped_model, metrics_df


def main():

    args = parse_inputs()

    ipath = Path(args.data_file)
    mpath = Path(args.metrics_file)
    npath = Path(args.network_file)

    if not ipath.is_file():
        raise IOError(f'Could not find input data file: {ipath}')

    if args.disable_gpu:
        tf.config.set_visible_devices([], 'GPU')

    if args.verbosity >= 2:
        tf.config.run_functions_eagerly(True)

    lpath = Path(args.log_file) if isinstance(args.log_file, str) else None
    setup_logging(logger, lpath, args.verbosity)
    logger.info(f'Starting Evidential BNN training script...')
    if args.verbosity >= 2:
        print_settings(logger, vars(args), 'Evidential training pipeline CLI settings:')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_tensorflow_pipeline_evidential(
        data=data,
        input_vars=args.input_var,
        output_vars=args.output_var,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        shuffle_seed=args.shuffle_seed,
        generalized_widths=args.generalized_node,
        specialized_depths=args.specialized_layer,
        specialized_widths=args.specialized_node,
        likelihood_weights=args.nll_weight,
        regularization_weights=args.reg_weight,
        learning_rate=args.learning_rate,
        decay_epoch=args.decay_epoch,
        decay_rate=args.decay_rate,
        verbosity=args.verbosity
    )

    metrics_dict.to_hdf(mpath, key='/data')
    logger.info(f' Metrics saved in {mpath}')

    trained_model.save(npath)
    logger.info(f' Network saved in {npath}')

    end_pipeline = time.perf_counter()

    logger.info(f'Pipeline completed! Total time: {(end_pipeline - start_pipeline):.4f} s')

    logger.info(f'Script completed!')


if __name__ == "__main__":
    main()

