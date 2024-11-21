import os
import argparse
import logging
import json
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from .train_pytorch_ncp import (
    launch_pytorch_pipeline_ncp
)
from ..utils.pipeline_tools import (
    setup_logging
)
from ..utils.helpers_pytorch import (
    default_dtype,
    default_device,
    get_device_info,
    set_device_parallelism,
    save_model
)

logger = logging.getLogger("tune_pytorch")


class HyperparameterTunerPytorchNCP():

    def __init__(
        self,
        data_path,
        input_var,
        output_var,
        settings,
        variables,
        network_path=None,
        metrics_path=None,
        log_path=None,
        verbosity=0
    ):

        if not isinstance(data_path, (str, Path)):
            raise TypeError('Data path argument must be a valid location! Aborting!')
        self.data_path = Path(data_path)
        self.settings = {'input_vars': input_var, 'output_vars': output_var, 'verbosity': verbosity}
        if isinstance(settings, dict):
            self.settings.update(settings)
        self.variables = {}
        if isinstance(variables, dict):
            for var, val in variables.items():
                if 'type' in val and 'lower_bound' in val and 'upper_bound' in val:
                    self.variables[var] = copy.deepcopy(val)
        self.npath = Path(network_path) if isinstance(network_path, (str, Path)) else None
        self.mpath = Path(metrics_path) if isinstance(metrics_path, (str, Path)) else None
        self.lpath = Path(log_path) if isinstance(log_path, (str, Path)) else None

    def __call__(self, trial):

        data = pd.read_hdf(self.data_path, key='/data')
        input_vars = self.settings.get('input_vars', [])
        output_vars = self.settings.get('output_vars', [])

        suggestion = self.sample_variables(trial)
        if 'minimum_performance' in suggestion:
            suggestion['minimum_performance'] = [suggestion['minimum_performance']] * len(output_vars)
        if 'num_generalized_layers' in suggestion:
            if 'num_generalized_nodes' in suggestion:
                suggestion['generalized_node'] = [suggestion['num_generalized_nodes']] * suggestion['num_generalized_layers']
            else:
                suggestion['generalized_node'] = [self.settings.get('generalized_node', [None])[0]] * suggestion['num_generalized_layers']
        elif 'num_generalized_nodes' in suggestion:
            suggestion['generalized_node'] = [suggestion['num_generalized_nodes']] * len(suggestion['generalized_node'])
        if 'ood_width' in suggestion:
            suggestion['ood_width'] = [suggestion['ood_width']] * len(output_vars)
        if 'epi_prior' in suggestion:
            suggestion['epi_prior'] = [suggestion['epi_prior']] * len(output_vars)
        if 'alea_prior' in suggestion:
            suggestion['alea_prior'] = [suggestion['alea_prior']] * len(output_vars)
        if 'nll_weight' in suggestion:
            suggestion['nll_weight'] = [suggestion['nll_weight']] * len(output_vars)
        if 'epi_weight' in suggestion:
            suggestion['epi_weight'] = [suggestion['epi_weight']] * len(output_vars)
        if 'alea_weight' in suggestion:
            suggestion['alea_weight'] = [suggestion['alea_weight']] * len(output_vars)
        if 'l1_reg_general' in suggestion:
            suggestion['l2_reg_general'] = 1.0 - suggestion['l1_reg_general']
            if suggestion['l2_reg_general'] < 0.0:
                suggestion['l2_reg_general'] = 0.0
        elif 'l2_reg_general' in suggestion:
            suggestion['l1_reg_general'] = 1.0 - suggestion['l2_reg_general']
            if suggestion['l1_reg_general'] < 0.0:
                suggestion['l1_reg_general'] = 0.0

        trained_model, metrics_dict = launch_pytorch_pipeline_ncp(
            data=data,
            input_vars=input_vars,
            output_vars=output_vars,
            validation_fraction=suggestion.get('validation_fraction', self.settings.get('validation_fraction', 0.1)),
            test_fraction=suggestion.get('test_fraction', self.settings.get('test_fraction', 0.1)),
            data_split_file=self.settings.get('data_split_file', None),
            max_epoch=suggestion.get('max_epoch', self.settings.get('max_epoch', 100)),
            batch_size=suggestion.get('batch_size', self.settings.get('batch_size', None)),
            early_stopping=suggestion.get('early_stopping', self.settings.get('early_stopping', None)),
            minimum_performance=suggestion.get('minimum_performance', self.settings.get('minimum_performance', None)),
            shuffle_seed=self.settings.get('shuffle_seed', None),
            sample_seed=self.settings.get('sample_seed', None),
            generalized_widths=suggestion.get('generalized_node', self.settings.get('generalized_node', None)),
            specialized_depths=suggestion.get('specialized_layer', self.settings.get('specialized_layer', None)),
            specialized_widths=suggestion.get('specialized_node', self.settings.get('specialized_node', None)),
            l1_regularization=suggestion.get('l1_reg_general', self.settings.get('l1_reg_general', 0.0)),
            l2_regularization=suggestion.get('l2_reg_general', self.settings.get('l2_reg_general', 0.0)),
            relative_regularization=suggestion.get('rel_reg_special', self.settings.get('rel_reg_special', 1.0)),
            ood_sampling_width=suggestion.get('ood_width', self.settings.get('ood_width', 0.1)),
            epistemic_priors=suggestion.get('epi_prior', self.settings.get('epi_prior', None)),
            aleatoric_priors=suggestion.get('alea_prior', self.settings.get('alea_prior', None)),
            distance_loss=self.settings.get('dist_loss_type', 'fisher_rao'),
            likelihood_weights=suggestion.get('nll_weight', self.settings.get('nll_weight', None)),
            epistemic_weights=suggestion.get('epi_weight', self.settings.get('epi_weight', None)),
            aleatoric_weights=suggestion.get('alea_weight', self.settings.get('alea_weight', None)),
            regularization_weights=suggestion.get('reg_weight', self.settings.get('reg_weight', 1.0)),
            learning_rate=suggestion.get('learning_rate', self.settings.get('learning_rate', 0.001)),
            decay_rate=suggestion.get('decay_rate', self.settings.get('decay_rate', 0.95)),
            decay_epoch=suggestion.get('decay_epoch', self.settings.get('decay_epoch', 20)),
            log_file=self.lpath,
            checkpoint_freq=self.settings.get('checkpoint_freq', 0),
            checkpoint_dir=self.settings.get('checkpoint_dir', None),
            save_initial_model=self.settings.get('save_initial', False),
            training_device=self.settings.get('training_device', default_device),
            verbosity=self.settings.get('verbosity', 0)
        )

        if self.mpath is not None:
            if not self.mpath.parent.is_dir():
                if not self.mpath.parent.exists():
                    self.mpath.parent.mkdir(parents=True)
                else:
                    raise IOError(f'Output directory path, {self.mpath.parent}, exists and is not a directory. Aborting!')
            metrics_dict.to_hdf(self.mpath, key='/data')
            logger.info(f' Metrics saved in {self.mpath}')

        if self.npath is not None:
            if not self.npath.parent.is_dir():
                if not self.npath.parent.exists():
                    self.npath.parent.mkdir(parents=True)
                else:
                    raise IOError(f'Output directory path, {self.npath.parent}, exists and is not a directory. Aborting!')
            save_model(trained_model, self.npath)
            logger.info(f' Network saved in {self.npath}')

        metric_name = 'valid_r2'
        performance = metrics_dict.loc[:, [var for var in metrics_dict if var.startswith(metric_name)]].min(axis=1).max()
        return performance

    def sample_variables(self, trial):
        suggestion = {}
        for var, specs in self.variables.items():
            if specs['type'] == 'int':
                log = specs.get('log', False)
                suggestion[var] = trial.suggest_int(var, specs['lower_bound'], specs['upper_bound'], log=log)
            elif specs['type'] == 'float':
                log = specs.get('log', False)
                suggestion[var] = trial.suggest_float(var, specs['lower_bound'], specs['upper_bound'], log=log)
        return suggestion


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Path and name of input HDF5 file containing training data set')
    parser.add_argument('--settings_file', metavar='path', type=str, required=True, help='Path and name of input JSON file containing network-specific settings')
    parser.add_argument('--tuning_file', metavar='path', type=str, required=True, help='Path and name of input JSON file containing tuning-specific settings')
    parser.add_argument('--database_file', metavar='path', type=str, required=True, help='Path and name of SQL database file containing optimization trials')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of output HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of output file to store trained model')
    parser.add_argument('--input_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to output log file where script related print outs will be stored')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def tune_pytorch_ncp_trial(
    data_path,
    input_vars,
    output_vars,
    settings_path,
    variable_path,
    database_path,
    network_path=None,
    metrics_path=None,
    log_path=None,
    verbosity=0
):

    with open(settings_path, 'r') as setfile:
        settings = json.load(setfile)
    with open(variable_path, 'r') as varfile:
        variables = json.load(varfile)
    study_name = variables.pop('study_name') if 'study_name' in variables else 'test'

    objective = HyperparameterTunerPytorchNCP(
        data_path,
        input_vars,
        output_vars,
        settings,
        variables,
        network_path=network_path,
        metrics_path=metrics_path,
        log_path=log_path,
        verbosity=verbosity
    )
    lock_obj = optuna.storages.JournalFileOpenLock(str(database_path.resolve()))
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(database_path, lock_obj=lock_obj))

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction='maximize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=1)

    trial = study.best_trial
    logger.info(f'Trial R2: {trial.value}')
    logger.info(f'Best hyperparameters: {trial.params}')

    return study.trials_dataframe()


def main():

    args = parse_inputs()

    data_path = Path(args.data_file)
    settings_path = Path(args.settings_file)
    variable_path = Path(args.tuning_file)
    database_path = Path(args.database_file)
    metrics_path = Path(args.metrics_file)
    network_path = Path(args.network_file)
    log_path = Path(args.log_file) if isinstance(args.log_file, str) else None

    run_id = os.environ.get('SLURM_ARRAY_TASK_ID', '')
    if run_id:
        metrics_path = metrics_path.with_stem(f'{metrics_path.stem}_{run_id}')
        network_path = network_path.with_stem(f'{network_path.stem}_{run_id}')
        log_path = log_path.with_stem(f'{log_path.stem}_{run_id}') if log_path is not None else None

    if not log_path.parent.is_dir():
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True)
        else:
            raise IOError(f'Output directory path, {log_path.parent}, exists and is not a directory. Aborting!')
    setup_logging(logger, log_path, args.verbosity)
    logger.info(f'Starting NCP BNN training script...')
    if args.verbosity >= 1:
        print_settings(logger, vars(args), 'NCP hyperparameter tuning pipeline CLI settings:')

    if variable_path.is_file():
        trial_data = tune_pytorch_ncp_trial(
            data_path,
            input_vars=args.input_var,
            output_vars=args.output_var,
            settings_path=settings_path,
            variable_path=variable_path,
            database_path=database_path,
            network_path=network_path,
            metrics_path=metrics_path,
            log_path=log_path,
            verbosity=args.verbosity
        )


if __name__ == "__main__":
    main()
