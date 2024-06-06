import argparse
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.distributions as tnd
from ..utils.pipeline_tools import setup_logging, print_settings, preprocess_data
from ..utils.helpers import mean_absolute_error, mean_squared_error
from ..utils.helpers_pytorch import create_data_loader, create_scheduled_adam_optimizer, create_model, create_loss_function, wrap_model

logger = logging.getLogger("train_pytorch")

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
    parser.add_argument('--sample_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for OOD sampling')
    parser.add_argument('--generalized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the generalized hidden layers')
    parser.add_argument('--specialized_layer', metavar='n', type=int, nargs='*', default=None, help='Number of specialized hidden layers, given for each output')
    parser.add_argument('--specialized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layers, sequential per output stack')
    parser.add_argument('--nll_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the NLL loss term')
    parser.add_argument('--reg_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to regularization loss term')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.98, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epoch', metavar='n', type=float, default=20, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device (not implemented)')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to log file where script related print outs will be stored')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def train_pytorch_evidential_epoch(
    model,
    optimizer,
    dataloader,
    loss_function,
    verbosity=0
):

    step_total_losses = []
    step_likelihood_losses = []
    step_regularization_losses = []

    # Training loop through minibatches - each loop pass is one step
    for nn, (feature_batch, target_batch) in enumerate(dataloader):

        # Set up training targets into a single large tensor
        target_values = torch.stack([target_batch, torch.zeros(target_batch.shape), torch.zeros(target_batch.shape), torch.zeros(target_batch.shape)], axis=1)
        batch_loss_targets = tf.stack([target_values, target_values], axis=2)
        n_outputs = batch_loss_targets.shape[-1]

        # Zero the gradients to avoid compounding over batches
        optimizer.zero_grad()

        # For mean data inputs, e.g. training data
        outputs = model(feature_batch)

        if verbosity >= 4:
            for ii in range(n_outputs):
                logger.debug(f'     gamma: {outputs[0, 0, ii]}')
                logger.debug(f'     nu: {outputs[0, 1, ii]}')
                logger.debug(f'     alpha: {outputs[0, 2, ii]}')
                logger.debug(f'     beta: {outputs[0, 3, ii]}')

        # Set up network predictions into equal shape tensor as training targets
        batch_loss_predictions = tf.stack([outputs, outputs], axis=2)

        # Compute total loss to be used in adjusting weights and biases
        if n_outputs == 1:
            batch_loss_targets = torch.squeeze(batch_loss_targets, dim=-1)
            batch_loss_predictions = torch.squeeze(batch_loss_predictions, dim=-1)
        step_total_loss = loss_function(batch_loss_targets, batch_loss_predictions)

        # Remaining loss terms purely for inspection purposes
        step_likelihood_loss = loss_function._calculate_likelihood_loss(
            torch.squeeze(torch.index_select(batch_loss_targets, dim=2, index=torch.tensor([0])), dim=2),
            torch.squeeze(torch.index_select(batch_loss_predictions, dim=2, index=torch.tensor([0])), dim=2)
        )
        step_regularization_loss = loss_function._calculate_regularization_loss(
            torch.squeeze(torch.index_select(batch_loss_targets, dim=2, index=torch.tensor([1])), dim=2),
            torch.squeeze(torch.index_select(batch_loss_predictions, dim=2, index=torch.tensor([1])), dim=2)
        )

        # Apply back-propagation
        step_total_loss.backward()
        optimizer.step()

        # Accumulate batch losses to determine epoch loss
        step_total_losses.append(torch.reshape(step_total_loss, shape=(-1, 1)))
        step_likelihood_losses.append(torch.reshape(step_likelihood_loss, shape=(-1, n_outputs)))
        step_regularization_losses.append(torch.reshape(step_regularization_loss, shape=(-1, n_outputs)))

        if verbosity >= 3:
            logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss.detach().numpy():.3f}')
            for ii in range(n_outputs):
                logger.debug(f'     Output {ii}: nll = {step_likelihood_loss.detach().numpy()[0, ii]:.3f}, reg = {step_regularization_loss.detach().numpy()[0, ii]:.3f}')

    epoch_total_loss = torch.sum(torch.cat(step_total_losses, dim=0), dim=0)
    epoch_likelihood_loss = torch.sum(torch.cat(step_likelihood_losses, dim=0), dim=0)
    epoch_regularization_loss = torch.sum(torch.cat(step_regularization_losses, dim=0), dim=0)

    return epoch_total_loss, epoch_likelihood_loss, epoch_regularization_loss


def train_pytorch_evidential(
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

    n_inputs = features_train.shape[-1]
    n_outputs = targets_train.shape[-1]
    train_length = features_train.shape[0]
    valid_length = features_valid.shape[0]

    if verbosity >= 1:
        logger.info(f' Number of inputs: {n_inputs}')
        logger.info(f' Number of outputs: {n_outputs}')
        logger.info(f' Training set size: {train_length}')
        logger.info(f' Validation set size: {valid_length}')

    # Create data loaders, including minibatching for training set
    train_data = (torch.tensor(features_train), torch.tensor(targets_train))
    valid_data = (torch.tensor(features_valid), torch.tensor(targets_valid))
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data)

    total_list = []
    nll_list = []
    reg_list = []
    mae_list = []
    mse_list = []

    # Training loop
    for epoch in range(max_epochs):

        model.train(True)

        # Training routine described in here
        epoch_total, epoch_nll, epoch_reg = train_pytorch_evidential_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            verbosity=verbosity
        )

        model.eval()

        total = epoch_total.detach().tolist()[0]
        nll = [np.nan] * n_outputs
        reg = [np.nan] * n_outputs
        mae = [np.nan] * n_outputs
        mse = [np.nan] * n_outputs
        with torch.no_grad():

            train_outputs = model(train_data[0])
            train_epistemic_avgs = train_outputs[:, 0, :]
            train_epistemic_stds = train_outputs[:, 1, :]
            train_aleatoric_rngs = train_outputs[:, 2, :]
            train_aleatoric_stds = train_outputs[:, 3, :]

            for ii in range(n_outputs):
                metric_targets = train_data[1][:, ii].detach().numpy()
                metric_results = train_epistemic_avgs[:, ii].detach().numpy()
                nll[ii] = epoch_nll.detach().tolist()[ii]
                reg[ii] = epoch_reg.detach().tolist()[ii]
                mae[ii] = mean_absolute_error(metric_targets, metric_results)
                mse[ii] = mean_squared_error(metric_targets, metric_results)

            if isinstance(patience, int):

                valid_outputs = model(valid_data[0])
                valid_epistemic_avgs = valid_outputs[:, 0, :]
                valid_epistemic_stds = valid_outputs[:, 1, :]
                valid_aleatoric_rngs = valid_outputs[:, 2, :]
                valid_aleatoric_stds = valid_outputs[:, 3, :]

        total_list.append(total)
        nll_list.append(nll)
        reg_list.append(reg)
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
                logger.debug(f'  Output {ii}: mse = {mse_list[-1][ii]:.3f}, mae = {mae_list[-1][ii]:.3f}, nll = {nll_list[-1][ii]:.3f}, reg = {reg_list[-1][ii]:.3f}')

    return total_list, mse_list, mae_list, nll_list, reg_list


def launch_pytorch_pipeline_evidential(
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
    disable_gpu=False,
    log_file=None,
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

    # Set up the NCP BNN model
    start_setup = time.perf_counter()
    n_inputs = features['train'].shape[-1]
    n_outputs = targets['train'].shape[-1]
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
    total_list, mse_list, mae_list, nll_list, reg_list = train_pytorch_evidential(
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

    # Save the trained model and training metrics
    start_out = time.perf_counter()
    total = np.array(total_list)
    mse = np.atleast_2d(mse_list)
    mae = np.atleast_2d(mae_list)
    nll = np.atleast_2d(nll_list)
    reg = np.atleast_2d(reg_list)
    metric_dict = {'total': total.flatten()}
    for ii in range(n_outputs):
        metric_dict[f'mse{ii}'] = mse[:, ii].flatten()
        metric_dict[f'mae{ii}'] = mae[:, ii].flatten()
        metric_dict[f'nll{ii}'] = nll[:, ii].flatten()
        metric_dict[f'reg{ii}'] = reg[:, ii].flatten()
    metrics = pd.DataFrame(data=metric_dict)
    descaled_model = wrap_model(model, features['scaler'], targets['scaler'])
    end_out = time.perf_counter()

    logger.info(f'Output configuration completed! Elapsed time: {(end_out - start_out):.4f} s')

    return descaled_model, metrics


def main():

    args = parse_inputs()

    ipath = Path(args.data_file)
    mpath = Path(args.metrics_file)
    npath = Path(args.network_file)

    if not ipath.is_file():
        raise IOError(f'Could not find input data file: {ipath}')

    lpath = Path(args.log_file) if isinstance(args.log_file, str) else None
    setup_logging(logger, lpath, args.verbosity)
    logger.info(f'Starting Evidential BNN training script...')
    if args.verbosity >= 2:
        print_settings(logger, vars(args), 'Evidential training pipeline CLI settings:')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_pytorch_pipeline_evidential(
        data=data,
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
        likelihood_weights=args.nll_weight,
        regularization_weights=args.reg_weight,
        learning_rate=args.learning_rate,
        decay_epoch=args.decay_epoch,
        decay_rate=args.decay_rate,
        verbosity=args.verbosity
    )

    metrics_dict.to_hdf(mpath, key='/data')
    logger.info(f' Metrics saved in {mpath}')

    torch.save(trained_model.state_dict(), npath)   # Needs the model class to reload
    logger.info(f' Network saved in {npath}')

    end_pipeline = time.perf_counter()

    logger.info(f'Pipeline completed! Total time: {(end_pipeline - start_pipeline):.4f} s')

    logger.info(f'Script completed!')


if __name__ == "__main__":
    main()

