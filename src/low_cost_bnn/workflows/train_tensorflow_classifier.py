import os
import argparse
import time
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from ..utils.pipeline_tools import setup_logging, print_settings
from ..utils.helpers_tensorflow import default_dtype, set_tf_logging_level, save_model
from .train_tensorflow_sngp import launch_tensorflow_pipeline_sngp

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


def launch_tensorflow_pipeline(
    data_file,
    input_vars,
    output_vars,
    settings_file,
    metrics_file,
    network_file,
    log_file=None,
    disable_gpu=False,
    verbosity=0
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

    if verbosity <= 4:
        set_tf_logging_level(logging.ERROR)

    if disable_gpu:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        tf.config.set_visible_devices([], 'GPU')

    if verbosity >= 2:
        tf.config.run_functions_eagerly(True)

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')
    specs = {}
    with open(spath, 'r') as jf:
        specs = json.load(jf)

    model_style = specs.get('style', None)
    metrics_df = None
    trained_model = None

    if model_style == 'sngp':

        trained_model, metrics_df = launch_tensorflow_pipeline_sngp(
            data=data,
            input_vars=input_vars,
            output_vars=output_vars,
            validation_fraction=specs.get('validation_fraction', 0.1),
            test_fraction=specs.get('test_fraction', 0.1),
            test_file=specs.get('test_file', None),
            max_epoch=specs.get('max_epoch', 10),
            batch_size=specs.get('batch_size', None),
            early_stopping=specs.get('early_stopping', None),
            shuffle_seed=specs.get('shuffle_seed', None),
            generalized_widths=specs.get('generalized_node', None),
            specialized_depths=specs.get('specialized_layer', None),
            specialized_widths=specs.get('specialized_node', None),
            spectral_normalization=specs.get('spec_norm_general', 0.9),
            relative_normalization=specs.get('rel_norm_special', 1.0),
            entropy_weights=specs.get('entropy_weight', None),
            regularization_weights=specs.get('reg_weight', 1.0),
            total_classes=specs.get('n_class', 1),
            learning_rate=specs.get('learning_rate', 0.001),
            decay_rate=specs.get('decay_rate', 0.98),
            decay_epoch=specs.get('decay_epoch', 50),
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
    status = launch_tensorflow_pipeline(
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

