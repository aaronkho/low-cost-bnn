import argparse
import time
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from ..utils.pipeline_tools import (
    setup_logging,
    print_settings
)
from ..utils.helpers_tensorflow import (
    default_dtype,
    default_device,
    save_model
)
from .train_tensorflow_ncp import launch_tensorflow_pipeline_ncp
from .train_tensorflow_evi import launch_tensorflow_pipeline_evidential

logger = logging.getLogger("train_tensorflow")


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Path and name of input HDF5 file containing training data set')
    parser.add_argument('--settings_file', metavar='path', type=str, required=True, help='Path and name of input JSON file containing network-specific settings')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of output HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of output file to store trained model')
    parser.add_argument('--input_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to output log file where script related print outs will be stored')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


def launch_tensorflow_regressor_pipeline(
    data_file,
    input_vars,
    output_vars,
    settings_file,
    metrics_file,
    network_file,
    log_file=None,
    disable_gpu=False,
    verbosity=0,
    **kwargs
):

    status = False

    settings = {
        'data_file': data_file,
        'input_vars': input_vars,
        'output_vars': output_vars,
        'settings_file': settings_file,
        'metrics_file': metrics_file,
        'network_file': network_file,
        'log_file': log_file,
        'disable_gpu': disable_gpu,
        'verbosity': verbosity,
    }

    lpath = Path(log_file) if isinstance(log_file, str) else None
    setup_logging(logger, lpath, verbosity)
    if verbosity >= 1:
        print_settings(logger, settings, 'General TensorFlow pipeline settings:')

    ipath = Path(data_file)
    spath = Path(settings_file)
    mpath = Path(metrics_file)
    npath = Path(network_file)

    if not ipath.is_file():
        raise IOError(f'Could not find input data file: {ipath}')

    if not spath.is_file():
        raise IOError(f'Could not find input settings file: {spath}')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')
    specs = {}
    with open(spath, 'r') as jf:
        specs = json.load(jf)
    specs['training_device'] = default_device if not disable_gpu else 'cpu'
    specs.update(kwargs)

    model_style = specs.get('style', None)
    metrics_df = None
    trained_model = None

    if model_style == 'ncp':

        trained_model, metrics_df = launch_tensorflow_pipeline_ncp(
            data=data,
            input_vars=input_vars,
            output_vars=output_vars,
            input_outlier_limit=specs.get('input_trim', None),
            output_outlier_limit=specs.get('output_trim', None),
            validation_fraction=specs.get('validation_fraction', 0.1),
            test_fraction=specs.get('test_fraction', 0.1),
            validation_data_file=specs.get('validation_data_file', None),
            data_split_file=specs.get('data_split_file', None),
            max_epoch=specs.get('max_epoch', 100),
            batch_size=specs.get('batch_size', None),
            early_stopping=specs.get('early_stopping', None),
            minimum_performance=specs.get('minimum_performance', None),
            shuffle_seed=specs.get('shuffle_seed', None),
            sample_seed=specs.get('sample_seed', None),
            generalized_widths=specs.get('generalized_node', None),
            specialized_depths=specs.get('specialized_layer', None),
            specialized_widths=specs.get('specialized_node', None),
            l1_regularization=specs.get('l1_reg_general', 0.0),
            l2_regularization=specs.get('l2_reg_general', 0.0),
            relative_regularization=specs.get('rel_reg_special', 1.0),
            ood_sampling_width=specs.get('ood_width', 1.0),
            epistemic_priors=specs.get('epi_prior', None),
            aleatoric_priors=specs.get('alea_prior', None),
            distance_loss=specs.get('dist_loss_type', 'fisher_rao'),
            likelihood_weights=specs.get('nll_weight', None),
            epistemic_weights=specs.get('epi_weight', None),
            aleatoric_weights=specs.get('alea_weight', None),
            regularization_weights=specs.get('reg_weight', 1.0),
            learning_rate=specs.get('learning_rate', 0.001),
            decay_rate=specs.get('decay_rate', 0.9),
            decay_epoch=specs.get('decay_epoch', 20),
            log_file=lpath,
            checkpoint_freq=specs.get('checkpoint_freq', 0),
            checkpoint_dir=specs.get('checkpoint_dir', None),
            save_initial_model=specs.get('save_initial', False),
            training_device=specs.get('training_device', default_device),
            verbosity=verbosity
        )
        status = True

    elif model_style == 'evidential':

        trained_model, metrics_df = launch_tensorflow_pipeline_evidential(
            data=data,
            input_vars=input_vars,
            output_vars=output_vars,
            input_outlier_limit=specs.get('input_trim', None),
            output_outlier_limit=specs.get('output_trim', None),
            validation_fraction=specs.get('validation_fraction', 0.1),
            test_fraction=specs.get('test_fraction', 0.1),
            validation_data_file=specs.get('validation_data_file', None),
            data_split_file=specs.get('data_split_file', None),
            max_epoch=specs.get('max_epoch', 100),
            batch_size=specs.get('batch_size', None),
            early_stopping=specs.get('early_stopping', None),
            minimum_performance=specs.get('minimum_performance', None),
            shuffle_seed=specs.get('shuffle_seed', None),
            generalized_widths=specs.get('generalized_node', None),
            specialized_depths=specs.get('specialized_layer', None),
            specialized_widths=specs.get('specialized_node', None),
            l1_regularization=specs.get('l1_reg_general', 0.0),
            l2_regularization=specs.get('l2_reg_general', 0.0),
            relative_regularization=specs.get('rel_reg_special', 1.0),
            likelihood_weights=specs.get('nll_weight', None),
            evidential_weights=specs.get('evi_weight', None),
            regularization_weights=specs.get('reg_weight', 1.0),
            learning_rate=specs.get('learning_rate', 0.001),
            decay_rate=specs.get('decay_rate', 0.9),
            decay_epoch=specs.get('decay_epoch', 20),
            log_file=lpath,
            checkpoint_freq=specs.get('checkpoint_freq', 0),
            checkpoint_dir=specs.get('checkpoint_dir', None),
            save_initial_model=specs.get('save_initial', False),
            training_device=specs.get('training_device', default_device),
            verbosity=verbosity
        )
        status = True

    if status and metrics_df is not None:
        if not mpath.parent.is_dir():
            if not mpath.parent.exists():
                mpath.parent.mkdir(parents=True)
            else:
                raise IOError(f'Output directory path, {mpath.parent}, exists and is not a directory. Aborting!')
        metrics_df.to_hdf(mpath, key='/data')
        logger.info(f' Metrics saved in {mpath}')

    if status and trained_model is not None:
        if not npath.parent.is_dir():
            if not npath.parent.exists():
                npath.parent.mkdir(parents=True)
            else:
                raise IOError(f'Output directory path, {npath.parent}, exists and is not a directory. Aborting!')
        save_model(trained_model, npath)
        logger.info(f' Network saved in {npath}')

    end_pipeline = time.perf_counter()

    logger.info(f'Pipeline completed! Total time: {(end_pipeline - start_pipeline):.4f} s')

    return status


def main():

    args = parse_inputs()
    status = launch_tensorflow_regressor_pipeline(
        data_file=args.data_file,
        input_vars=args.input_var,
        output_vars=args.output_var,
        settings_file=args.settings_file,
        metrics_file=args.metrics_file,
        network_file=args.network_file,
        log_file=args.log_file,
        disable_gpu=args.disable_gpu,
        verbosity=args.verbosity
    )
    if status:
        print(f'TensorFlow training script completed successfully!')
    else:
        print(f'Unexpected error in TensorFlow training script...')


if __name__ == "__main__":
    main()

