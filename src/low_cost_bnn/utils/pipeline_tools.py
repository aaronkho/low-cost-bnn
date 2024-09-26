import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .helpers import create_scaler, split


def setup_logging(logger, log_path=None, verbosity=0):

    logger.propagate = False

    formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    logger.setLevel(logging.INFO)
    if verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():

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
    data_split_savepath,
    shuffle=True,
    seed=None,
    trim_feature_outliers=None,
    trim_target_outliers=None,
    scale_features=True,
    scale_targets=True,
    logger=None,
    verbosity=0
):

    ml_vars = []
    ml_vars.extend(feature_vars)
    ml_vars.extend(target_vars)
    ml_data = data.loc[:, ml_vars].astype(np.float64)

    outlier_mask = np.isfinite(ml_data.iloc[:, 0])
    if isinstance(trim_feature_outliers, (float, int)):
        feature_mean = ml_data.loc[:, feature_vars].mean(axis=0)
        feature_stdev = ml_data.loc[:, feature_vars].std(axis=0, ddof=0)
        for var in feature_vars:
            if var in feature_mean and var in feature_stdev:
                outlier_mask &= (((ml_data.loc[:, var] - feature_mean[var]) / feature_stdev[var]).abs() < np.abs(trim_feature_outliers))
    if isinstance(trim_target_outliers, (float, int)):
        target_mean = ml_data.loc[:, target_vars].mean(axis=0)
        target_stdev = ml_data.loc[:, target_vars].std(axis=0, ddof=0)
        for var in target_vars:
            if var in target_mean and var in target_stdev:
                outlier_mask &= (((ml_data.loc[:, var] - target_mean[var]) / target_stdev[var]).abs() < np.abs(trim_target_outliers))
    ml_data = ml_data.loc[outlier_mask, :]

    feature_scaler = create_scaler(ml_data.loc[:, feature_vars]) if scale_features else None
    target_scaler = create_scaler(ml_data.loc[:, target_vars]) if scale_targets else None

    first_split = validation_fraction + test_fraction
    second_split = test_fraction / first_split
    train_data, split_data = split(ml_data, first_split, shuffle=shuffle, seed=seed)
    val_data, test_data = split(split_data, second_split, shuffle=shuffle, seed=seed)
    test_data = test_data.sort_index()

    # Saving data split indices for post-processing reconstruction
    index_length = len(ml_data)
    index_df = pd.DataFrame(data={'dataset': [0] * index_length}, index=ml_data.index.values)
    index_df.loc[index_df.index.isin(val_data.index), 'dataset'] = 1
    index_df.loc[index_df.index.isin(test_data.index), 'dataset'] = 2

    if isinstance(data_split_savepath, Path):
        if not data_split_savepath.exists():
            if not data_split_savepath.parent.is_dir():
                data_split_savepath.parent.mkdir(parents=True)
            index_df.to_hdf(data_split_savepath,key='/data',mode='w')
        elif logger is not None:
            logger.warning(f'Indices for data split save file, {data_split_savepath}, already exists! Aborting save...')

    feature_train = train_data.loc[:, feature_vars].to_numpy()
    feature_val = val_data.loc[:, feature_vars].to_numpy()
    feature_test = test_data.loc[:, feature_vars].to_numpy()
    if scale_features:
        feature_train = feature_scaler.transform(train_data.loc[:, feature_vars])
        feature_val = feature_scaler.transform(val_data.loc[:, feature_vars])
        feature_test = feature_scaler.transform(test_data.loc[:, feature_vars])

    target_train = train_data.loc[:, target_vars].to_numpy()
    target_val = val_data.loc[:, target_vars].to_numpy()
    target_test = test_data.loc[:, target_vars].to_numpy()
    if scale_targets:
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

