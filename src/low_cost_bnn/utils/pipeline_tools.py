import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .helpers import create_scaler, split


def setup_logging(logger, log_path=None, verbosity=0):

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


def print_settings(logger, settings, header=None):
    if isinstance(header, str):
        logger.debug(header)
    for key, val in settings.items():
        logger.debug(f'  {key}: {val}')


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
        'original_train': np.atleast_2d(train_data.loc[:, feature_vars].to_numpy()),
        'original_validation': np.atleast_2d(val_data.loc[:, feature_vars].to_numpy()),
        'original_test': np.atleast_2d(test_data.loc[:, feature_vars].to_numpy()),
        'train': np.atleast_2d(feature_train),
        'validation': np.atleast_2d(feature_val),
        'test': np.atleast_2d(feature_test),
        'scaler': feature_scaler,
    }
    targets = {
        'names': target_vars,
        'original_train': np.atleast_2d(train_data.loc[:, target_vars].to_numpy()),
        'original_validation': np.atleast_2d(val_data.loc[:, target_vars].to_numpy()),
        'original_test': np.atleast_2d(test_data.loc[:, target_vars].to_numpy()),
        'train': np.atleast_2d(target_train),
        'validation': np.atleast_2d(target_val),
        'test': np.atleast_2d(target_test),
        'scaler': target_scaler,
    }

    return features, targets

