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
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device (not implemented)')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to log file where script related print outs will be stored')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def train_pytorch_ncp_epoch(
    model,
    optimizer,
    dataloader,
    loss_function,
    ood_sigmas,
    ood_seed=None,
    verbosity=0
):

    step_total_losses = []
    step_likelihood_losses = []
    step_epistemic_losses = []
    step_aleatoric_losses = []

    gen = torch.Generator()
    if isinstance(ood_seed, int):
        gen.manual_seed(ood_seed)

    # Training loop through minibatches - each loop pass is one step
    for nn, (feature_batch, target_batch, epistemic_sigma_batch, aleatoric_sigma_batch) in enumerate(dataloader):

        # Set up training targets into a single large tensor
        target_values = torch.stack([target_batch, torch.zeros(target_batch.shape)], dim=1)
        epistemic_prior_moments = torch.stack([target_batch, epistemic_sigma_batch], dim=1)
        aleatoric_prior_moments = torch.stack([target_batch, aleatoric_sigma_batch], dim=1)
        batch_loss_targets = torch.stack([target_values, epistemic_prior_moments, aleatoric_prior_moments], dim=2)
        n_outputs = batch_loss_targets.shape[-1]

        # Generate random OOD data from training data
        ood_feature_batch = torch.zeros(feature_batch.shape)
        for jj in range(feature_batch.shape[1]):
            ood = torch.normal(feature_batch[:, jj], ood_sigmas[jj], generator=gen)
            ood_feature_batch[:, jj] = ood

        # Zero the gradients to avoid compounding over batches
        optimizer.zero_grad()

        # For mean data inputs, e.g. training data
        mean_outputs = model(feature_batch)
        mean_epistemic_avgs = torch.squeeze(torch.index_select(mean_outputs, dim=1, index=torch.tensor([0])), dim=1)
        mean_epistemic_stds = torch.squeeze(torch.index_select(mean_outputs, dim=1, index=torch.tensor([1])), dim=1)
        mean_aleatoric_rngs = torch.squeeze(torch.index_select(mean_outputs, dim=1, index=torch.tensor([2])), dim=1)
        mean_aleatoric_stds = torch.squeeze(torch.index_select(mean_outputs, dim=1, index=torch.tensor([3])), dim=1)

        # For OOD data inputs
        ood_outputs = model(ood_feature_batch)
        ood_epistemic_avgs = torch.squeeze(torch.index_select(ood_outputs, dim=1, index=torch.tensor([0])), dim=1)
        ood_epistemic_stds = torch.squeeze(torch.index_select(ood_outputs, dim=1, index=torch.tensor([1])), dim=1)
        ood_aleatoric_rngs = torch.squeeze(torch.index_select(ood_outputs, dim=1, index=torch.tensor([2])), dim=1)
        ood_aleatoric_stds = torch.squeeze(torch.index_select(ood_outputs, dim=1, index=torch.tensor([3])), dim=1)

        if verbosity >= 4:
            for ii in range(n_outputs):
                logger.debug(f'     In-dist model: {mean_epistemic_avgs[0, ii].detach().numpy()}, {mean_epistemic_stds[0, ii].detach().numpy()}')
                logger.debug(f'     In-dist noise: {mean_aleatoric_rngs[0, ii].detach().numpy()}, {mean_aleatoric_stds[0, ii].detach().numpy()}')
                logger.debug(f'     Out-of-dist model: {ood_epistemic_avgs[0, ii].detach().numpy()}, {ood_epistemic_stds[0, ii].detach().numpy()}')
                logger.debug(f'     Out-of-dist noise: {ood_aleatoric_rngs[0, ii].detach().numpy()}, {ood_aleatoric_stds[0, ii].detach().numpy()}')

        # Set up network predictions into equal shape tensor as training targets
        prediction_distributions = torch.stack([mean_aleatoric_rngs, mean_aleatoric_stds], dim=1)
        epistemic_posterior_moments = torch.stack([ood_epistemic_avgs, ood_epistemic_stds], dim=1)
        aleatoric_posterior_moments = torch.stack([ood_aleatoric_rngs, ood_aleatoric_stds], dim=1)
        batch_loss_predictions = torch.stack([prediction_distributions, epistemic_posterior_moments, aleatoric_posterior_moments], dim=2)

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
        step_epistemic_loss = loss_function._calculate_model_divergence_loss(
            torch.squeeze(torch.index_select(batch_loss_targets, dim=2, index=torch.tensor([1])), dim=2),
            torch.squeeze(torch.index_select(batch_loss_predictions, dim=2, index=torch.tensor([1])), dim=2)
        )
        step_aleatoric_loss = loss_function._calculate_noise_divergence_loss(
            torch.squeeze(torch.index_select(batch_loss_targets, dim=2, index=torch.tensor([2])), dim=2),
            torch.squeeze(torch.index_select(batch_loss_predictions, dim=2, index=torch.tensor([2])), dim=2)
        )

        # Apply back-propagation
        step_total_loss.backward()
        optimizer.step()

        # Accumulate batch losses to determine epoch loss
        step_total_losses.append(torch.reshape(step_total_loss, shape=(-1, 1)))
        step_likelihood_losses.append(torch.reshape(step_likelihood_loss, shape=(-1, n_outputs)))
        step_epistemic_losses.append(torch.reshape(step_epistemic_loss, shape=(-1, n_outputs)))
        step_aleatoric_losses.append(torch.reshape(step_aleatoric_loss, shape=(-1, n_outputs)))

        if verbosity >= 3:
            logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss.detach().numpy():.3f}')
            for ii in range(n_outputs):
                logger.debug(f'     Output {ii}: nll = {step_likelihood_loss.detach().numpy()[0, ii]:.3f}, epi = {step_epistemic_loss.detach().numpy()[0, ii]:.3f}, alea = {step_aleatoric_loss.detach().numpy()[0, ii]:.3f}')

    epoch_total_loss = torch.sum(torch.cat(step_total_losses, dim=0), dim=0)
    epoch_likelihood_loss = torch.sum(torch.cat(step_likelihood_losses, dim=0), dim=0)
    epoch_epistemic_loss = torch.sum(torch.cat(step_epistemic_losses, dim=0), dim=0)
    epoch_aleatoric_loss = torch.sum(torch.cat(step_aleatoric_losses, dim=0), dim=0)

    return epoch_total_loss, epoch_likelihood_loss, epoch_epistemic_loss, epoch_aleatoric_loss


def train_pytorch_ncp(
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
    train_data = (torch.tensor(features_train), torch.tensor(targets_train), torch.tensor(epi_priors), torch.tensor(alea_priors))
    valid_data = (torch.tensor(features_valid), torch.tensor(targets_valid))
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data)

    total_list = []
    nll_list = []
    epi_list = []
    alea_list = []
    mae_list = []
    mse_list = []

    # Training loop
    for epoch in range(max_epochs):

        model.train(True)

        # Training routine described in here
        epoch_total, epoch_nll, epoch_epi, epoch_alea = train_pytorch_ncp_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            ood_sigma,
            ood_seed=None,
            verbosity=verbosity
        )

        model.eval()

        total = epoch_total.detach().tolist()[0]
        nll = [np.nan] * n_outputs
        epi = [np.nan] * n_outputs
        alea = [np.nan] * n_outputs
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
                epi[ii] = epoch_epi.detach().tolist()[ii]
                alea[ii] = epoch_alea.detach().tolist()[ii]
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

    return total_list, mse_list, mae_list, nll_list, epi_list, alea_list


def launch_pytorch_pipeline_ncp(
    data,
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
    }

    if verbosity >= 2:
        print_settings(logger, settings, 'NCP model and training settings:')

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

    logger.info(f'Pre-processing completed! Elpased time: {(end_preprocess - start_preprocess):.4f} s')

    # Set up the NCP BNN model
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
    total_list, mse_list, mae_list, nll_list, epi_list, alea_list = train_pytorch_ncp(
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
    start_out = time.perf_counter()
    total = np.array(total_list)
    mse = np.atleast_2d(mse_list)
    mae = np.atleast_2d(mae_list)
    nll = np.atleast_2d(nll_list)
    epi = np.atleast_2d(epi_list)
    alea = np.atleast_2d(alea_list)
    metric_dict = {'total': total.flatten()}
    for ii in range(n_outputs):
        metric_dict[f'mse{ii}'] = mse[:, ii].flatten()
        metric_dict[f'mae{ii}'] = mae[:, ii].flatten()
        metric_dict[f'nll{ii}'] = nll[:, ii].flatten()
        metric_dict[f'epi{ii}'] = epi[:, ii].flatten()
        metric_dict[f'alea{ii}'] = alea[:, ii].flatten()
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
    logger.info(f'Starting NCP-BNN training script...')
    if args.verbosity >= 2:
        print_settings(logger, vars(args), 'NCP training pipeline settings...')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_pytorch_pipeline_ncp(
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
        ood_sampling_width=args.ood_width,
        epistemic_priors=args.epi_prior,
        aleatoric_priors=args.alea_prior,
        likelihood_weights=args.nll_weight,
        epistemic_weights=args.epi_weight,
        aleatoric_weights=args.alea_weight,
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

