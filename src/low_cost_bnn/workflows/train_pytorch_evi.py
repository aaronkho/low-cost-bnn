import argparse
import time
import copy
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.distributions as tnd
from ..utils.pipeline_tools import setup_logging, print_settings, preprocess_data
from ..utils.helpers import mean_absolute_error, mean_squared_error, fbeta_score, adjusted_r2_score
from ..utils.helpers_pytorch import default_dtype, create_data_loader, create_scheduled_adam_optimizer, create_regressor_model, create_regressor_loss_function, wrap_regressor_model, save_model

logger = logging.getLogger("train_pytorch")


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
    parser.add_argument('--minimum_performance', metavar='val', type=float, default=None, help='Set minimum value in adjusted R-squared before early stopping is activated')
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
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device (not implemented)')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to output log file where script related print outs will be stored')
    parser.add_argument('--checkpoint_freq', metavar='n', type=int, default=0, help='Number of epochs between saves of model checkpoint')
    parser.add_argument('--checkpoint_dir', metavar='path', type=str, default=None, help='Optional path to directory where checkpoints will be saved')
    parser.add_argument('--save_initial', default=False, action='store_true', help='Toggle on saving of initialized model before any training, for debugging')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def train_pytorch_evidential_epoch(
    model,
    optimizer,
    dataloader,
    loss_function,
    reg_weight,
    training=True,
    dataset_length=None,
    verbosity=0
):

    step_total_losses = []
    step_regularization_losses = []
    step_likelihood_losses = []
    step_evidential_losses = []

    model.train()
    if not training:
        model.eval()

    if dataset_length is None:
        dataset_length = len(dataloader.dataset)
    dataset_size = torch.tensor([dataset_length], dtype=default_dtype)

    # Training loop through minibatches - each loop pass is one step
    for nn, (feature_batch, target_batch) in enumerate(dataloader):

        batch_size = torch.tensor([feature_batch.shape[0]], dtype=default_dtype)

        # Set up training targets into a single large tensor
        target_values = torch.stack([target_batch, torch.zeros(target_batch.shape, dtype=default_dtype), torch.zeros(target_batch.shape, dtype=default_dtype), torch.zeros(target_batch.shape, dtype=default_dtype)], dim=1)
        batch_loss_targets = torch.stack([target_values, target_values], dim=2)
        n_outputs = batch_loss_targets.shape[-1]

        # Zero the gradients to avoid compounding over batches
        if training:
            optimizer.zero_grad()

        # For mean data inputs, e.g. training data
        outputs = model(feature_batch)

        if training and verbosity >= 4:
            for ii in range(n_outputs):
                logger.debug(f'     gamma: {outputs[0, 0, ii]}')
                logger.debug(f'     nu: {outputs[0, 1, ii]}')
                logger.debug(f'     alpha: {outputs[0, 2, ii]}')
                logger.debug(f'     beta: {outputs[0, 3, ii]}')

        # Set up network predictions into equal shape tensor as training targets
        batch_loss_predictions = torch.stack([outputs, outputs], dim=2)

        # Acquire regularization loss after evaluation of network
        model_metrics = model.get_metrics_result()
        step_regularization_loss = torch.tensor([0.0], dtype=default_dtype)
        if 'regularization_loss' in model_metrics:
            weight = torch.tensor([reg_weight], dtype=default_dtype)
            step_regularization_loss = weight * model_metrics['regularization_loss']
        # Regularization loss is invariant on batch size, but this improves comparative context in metrics
        step_regularization_loss = step_regularization_loss * batch_size / dataset_size

        # Compute total loss to be used in adjusting weights and biases
        if n_outputs == 1:
            batch_loss_targets = torch.squeeze(batch_loss_targets, dim=-1)
            batch_loss_predictions = torch.squeeze(batch_loss_predictions, dim=-1)
        step_total_loss = loss_function(batch_loss_targets, batch_loss_predictions)
        step_total_loss = step_total_loss + step_regularization_loss
        adjusted_step_total_loss = step_total_loss / batch_size

        # Remaining loss terms purely for inspection purposes
        step_likelihood_loss = loss_function._calculate_likelihood_loss(
            torch.squeeze(torch.index_select(batch_loss_targets, dim=2, index=torch.tensor([0])), dim=2),
            torch.squeeze(torch.index_select(batch_loss_predictions, dim=2, index=torch.tensor([0])), dim=2)
        )
        step_evidential_loss = loss_function._calculate_evidential_loss(
            torch.squeeze(torch.index_select(batch_loss_targets, dim=2, index=torch.tensor([1])), dim=2),
            torch.squeeze(torch.index_select(batch_loss_predictions, dim=2, index=torch.tensor([1])), dim=2)
        )

        # Apply back-propagation
        if training:
            adjusted_step_total_loss.backward()
            optimizer.step()

        # Accumulate batch losses to determine epoch loss
        step_total_losses.append(torch.reshape(step_total_loss, shape=(-1, 1)))
        step_regularization_losses.append(torch.reshape(step_regularization_loss, shape=(-1, 1)))
        step_likelihood_losses.append(torch.reshape(step_likelihood_loss, shape=(-1, n_outputs)))
        step_evidential_losses.append(torch.reshape(step_regularization_loss, shape=(-1, n_outputs)))

        if verbosity >= 3:
            if training:
                logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss.detach().numpy():.3f}, reg = {step_regularization_loss.detach().numpy():.3f}')
                for ii in range(n_outputs):
                    logger.debug(f'     Output {ii}: nll = {step_likelihood_loss.detach().numpy()[0, ii]:.3f}, evi = {step_evidential_loss.detach().numpy()[0, ii]:.3f}')
            else:
                logger.debug(f'  - Validation: total = {step_total_loss.detach().numpy():.3f}, reg = {step_regularization_loss.detach().numpy():.3f}')
                for ii in range(n_outputs):
                    logger.debug(f'     Output {ii}: nll = {step_likelihood_loss.detach().numpy()[0, ii]:.3f}, evi = {step_evidential_loss.detach().numpy()[0, ii]:.3f}')

    epoch_total_loss = torch.sum(torch.cat(step_total_losses, dim=0), dim=0)
    epoch_regularization_loss = torch.sum(torch.cat(step_regularization_losses, dim=0), dim=0)
    epoch_likelihood_loss = torch.sum(torch.cat(step_likelihood_losses, dim=0), dim=0)
    epoch_evidential_loss = torch.sum(torch.cat(step_evidential_losses, dim=0), dim=0)

    if not training:
        model.train()

    return epoch_total_loss, epoch_regularization_loss, epoch_likelihood_loss, epoch_evidential_loss


def train_pytorch_evidential(
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
    r2_minimum=None,
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
    r2_threshold = float(r2_minimum) if isinstance(r2_minimum, (float, int)) else -1.0

    if verbosity >= 2:
        logger.info(f' Number of inputs: {n_inputs}')
        logger.info(f' Number of outputs: {n_outputs}')
        logger.info(f' Training set size: {train_length}')
        logger.info(f' Validation set size: {valid_length}')

    # Create data loaders, including minibatching for training set
    train_data = (
        torch.tensor(features_train, dtype=default_dtype),
        torch.tensor(targets_train, dtype=default_dtype)
    )
    valid_data = (
        torch.tensor(features_valid, dtype=default_dtype),
        torch.tensor(targets_valid, dtype=default_dtype)
    )
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data, batch_size=valid_length)

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
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(model.state_dict())
    best_model.eval()

    # Training loop
    stop_requested = False
    threshold_surpassed = False
    for epoch in range(max_epochs):

        # Training routine described in here
        epoch_total, epoch_reg, epoch_nll, epoch_evi = train_pytorch_evidential_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            reg_weight,
            training=True,
            dataset_length=train_length,
            verbosity=verbosity
        )

        total_train = epoch_total.detach().tolist()[0] / train_length
        reg_train = epoch_reg.detach().tolist()[0] / train_length
        nll_train = [np.nan] * n_outputs
        evi_train = [np.nan] * n_outputs
        r2_train = [np.nan] * n_outputs
        mae_train = [np.nan] * n_outputs
        mse_train = [np.nan] * n_outputs

        model.eval()
        with torch.no_grad():

            # Evaluate model with full training data set for performance tracking
            train_outputs = model(train_data[0])
            train_means = torch.squeeze(torch.index_select(train_outputs, dim=1, index=torch.tensor([0])), dim=1)

            for ii in range(n_outputs):
                metric_targets = np.atleast_2d(train_data[1][:, ii].detach().numpy()).T
                metric_results = np.atleast_2d(train_means[:, ii].detach().numpy()).T
                nll_train[ii] = epoch_nll.detach().tolist()[ii] / train_length
                evi_train[ii] = epoch_evi.detach().tolist()[ii] / train_length
                r2_train[ii] = adjusted_r2_score(metric_targets, metric_results, nreg=n_inputs)[0]
                mae_train[ii] = mean_absolute_error(metric_targets, metric_results)[0]
                mse_train[ii] = mean_squared_error(metric_targets, metric_results)[0]

        total_train_list.append(total_train)
        reg_train_list.append(reg_train)
        nll_train_list.append(nll_train)
        evi_train_list.append(evi_train)
        r2_train_list.append(r2_train)
        mae_train_list.append(mae_train)
        mse_train_list.append(mse_train)

        model.train()

        # Reuse training routine to evaluate validation data
        valid_total, valid_reg, valid_nll, valid_evi = train_pytorch_evidential_epoch(
            model,
            optimizer,
            valid_loader,
            loss_function,
            reg_weight,
            training=False,
            dataset_length=valid_length,
            verbosity=verbosity
        )

        total_valid = valid_total.detach().tolist()[0] / valid_length
        reg_valid = valid_reg.detach().tolist()[0] / train_length  # Invariant to batch size, needed for comparison
        nll_valid = [np.nan] * n_outputs
        evi_valid = [np.nan] * n_outputs
        r2_valid = [np.nan] * n_outputs
        mae_valid = [np.nan] * n_outputs
        mse_valid = [np.nan] * n_outputs

        model.eval()
        with torch.no_grad():

            # Evaluate model with validation data set for performance tracking
            valid_outputs = model(valid_data[0])
            valid_means = torch.squeeze(torch.index_select(valid_outputs, dim=1, index=torch.tensor([0])), dim=1)

            for ii in range(n_outputs):
                metric_targets = np.atleast_2d(valid_data[1][:, ii].detach().numpy()).T
                metric_results = np.atleast_2d(valid_means[:, ii].detach().numpy()).T
                nll_valid[ii] = valid_nll.detach().tolist()[ii] / valid_length
                evi_valid[ii] = valid_evi.detach().tolist()[ii] / valid_length
                r2_valid[ii] = adjusted_r2_score(metric_targets, metric_results, nreg=n_inputs)[0]
                mae_valid[ii] = mean_absolute_error(metric_targets, metric_results)[0]
                mse_valid[ii] = mean_squared_error(metric_targets, metric_results)[0]

        total_valid_list.append(total_valid)
        reg_valid_list.append(reg_valid)
        nll_valid_list.append(nll_valid)
        evi_valid_list.append(evi_valid)
        r2_valid_list.append(r2_valid)
        mae_valid_list.append(mae_valid)
        mse_valid_list.append(mse_valid)

        model.train()

        # Enable early stopping routine if minimum performance threshold is met
        if not threshold_surpassed:
            if not np.isfinite(np.nanmean(r2_valid_list[-1])):
                threshold_surpassed = True
                logger.warning(f'Adjusted R-squared metric is NaN, enabling early stopping to prevent large computational waste...')
            if np.nanmean(r2_valid_list[-1]) >= r2_threshold:
                threshold_surpassed = True
                if r2_threshold >= 0.0:
                    logger.info(f'Requested minimum performance of {r2_threshold:.5f} exceeded at epoch {epoch + 1}')

        # Save model into output container if it is the best so far
        if threshold_surpassed:
            if best_validation_loss is None:
                best_validation_loss = total_valid_list[-1] + improve_tol + 1.0e-3
            n_no_improve = n_no_improve + 1 if best_validation_loss < (total_valid_list[-1] + improve_tol) else 0
            if n_no_improve == 0:
                best_validation_loss = total_valid_list[-1]
                best_model.load_state_dict(model.state_dict())

        # Request training stop if early stopping is enabled
        if isinstance(patience, int) and patience > 0 and n_no_improve >= patience:
            stop_requested = True

        print_per_epochs = 100
        if verbosity >= 2:
            print_per_epochs = 10
        if verbosity >= 3:
            print_per_epochs = 1
        if (epoch + 1) % print_per_epochs == 0:
            logger.info(f' Epoch {epoch + 1}: total_train = {total_train_list[-1]:.3f}, total_valid = {total_valid_list[-1]:.3f}')
            logger.info(f'       {epoch + 1}: reg_train = {reg_train_list[-1]:.3f}, reg_valid = {reg_valid_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.debug(f'  Train Output {ii}: r2 = {r2_train_list[-1][ii]:.3f}, mse = {mse_train_list[-1][ii]:.3f}, mae = {mae_train_list[-1][ii]:.3f}, nll = {nll_train_list[-1][ii]:.3f}, evi = {evi_train_list[-1][ii]:.3f}')
                logger.debug(f'  Valid Output {ii}: r2 = {r2_valid_list[-1][ii]:.3f}, mse = {mse_valid_list[-1][ii]:.3f}, mae = {mae_valid_list[-1][ii]:.3f}, nll = {nll_valid_list[-1][ii]:.3f}, evi = {evi_valid_list[-1][ii]:.3f}')

        # Model Checkpoint
        # ------------------------------------------------
        if checkpoint_path is not None and checkpoint_freq > 0:
            if (epoch + 1) % checkpoint_freq == 0:
                check_path = checkpoint_path / f'checkpoint_model_epoch{epoch+1}.pt'
                checkpoint_model = copy.deepcopy(model)
                checkpoint_model.load_state_dict(model.state_dict())
                checkpoint_model.eval()
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

        # Exit training loop early if requested
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


def launch_pytorch_pipeline_evidential(
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
    disable_gpu=False,
    log_file=None,
    checkpoint_freq=0,
    checkpoint_dir=None,
    save_initial_model=False,
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
        'checkpoint_freq': checkpoint_freq,
        'checkpoint_dir': checkpoint_dir,
        'save_initial_model': save_initial_model,
    }

    if verbosity >= 1:
        print_settings(logger, settings, 'Evidential model and training settings:')

    # Set up the required data sets
    start_preprocess = time.perf_counter()
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
            initpath = checkpoint_path / 'checkpoint_model_initial.pt'
            initial_model = copy.deepcopy(model)
            initial_model.load_state_dict(model.state_dict())
            initial_model.eval()
            if 'scaler' in features and features['scaler'] is not None and 'scaler' in targets and targets['scaler'] is not None:
                initial_model = wrap_regressor_model(initial_model, features['scaler'], targets['scaler'])
            save_model(initial_model, initpath)
        else:
            logger.warning(f'Requested initial model save cannot be made due to invalid checkpoint directory, {checkpoint_path}. Initial save will be skipped!')
            checkpoint_path = None
    best_model, metrics = train_pytorch_evidential(
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
        r2_minimum=minimum_performance,
        checkpoint_freq=checkpoint_freq,
        checkpoint_path=checkpoint_path,
        features_scaler=features['scaler'],
        targets_scaler=targets['scaler'],
        verbosity=verbosity
    )
    end_train = time.perf_counter()

    logger.info(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

    # Save the trained model and training metrics
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
        inputs = torch.zeros([1, best_model.n_inputs], dtype=default_dtype)
        outputs = model(inputs)
        logger.debug(f'  Sample output at origin:')
        logger.debug(f'{outputs}')

    return wrapped_model, metrics_df


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
    if args.verbosity >= 1:
        print_settings(logger, vars(args), 'Evidential training pipeline CLI settings:')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_pytorch_pipeline_evidential(
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
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        save_initial_model=args.save_initial,
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

