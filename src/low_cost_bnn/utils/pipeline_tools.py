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
    validation_loadpath=None,
    data_split_savepath=None,
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
    original_data_index = ml_data.index.values
    outlier_data = ml_data.loc[~outlier_mask, :]
    ml_data = ml_data.loc[outlier_mask, :]

    feature_scaler = create_scaler(ml_data.loc[:, feature_vars]) if scale_features else None
    target_scaler = create_scaler(ml_data.loc[:, target_vars]) if scale_targets else None

    val_data = pd.DataFrame(columns=ml_data.columns)
    test_data = pd.DataFrame(columns=ml_data.columns)
    if isinstance(validation_loadpath, Path):
        if validation_loadpath.exists():
            validation_fraction = 0.0
            test_fraction = 0.0
            val_data = pd.read_hdf(validation_loadpath, key='/data')
            logger.info(f'Loaded fixed validation set from file, {validation_loadpath}.')
        else:
            logger.warning(f'Specified validation set load file, {validation_loadpath}, does not exists! Skipping load...')

    first_split = validation_fraction + test_fraction
    if first_split > 0.0:
        second_split = test_fraction / first_split
        train_data, split_data = split(ml_data, first_split, shuffle=shuffle, seed=seed)
        if second_split > 0.0:
            val_data, test_data = split(split_data, second_split, shuffle=shuffle, seed=seed)
        else:
            val_data = split_data
    else:
        train_data = ml_data
    test_data = test_data.sort_index()

    # Saving data split indices for post-processing reconstruction
    index_length = len(original_data_index)
    index_df = pd.DataFrame(data={'dataset': [0] * index_length}, index=original_data_index)
    index_df.loc[index_df.index.isin(val_data.index), 'dataset'] = 1
    index_df.loc[index_df.index.isin(test_data.index), 'dataset'] = 2
    index_df.loc[index_df.index.isin(outlier_data.index), 'dataset'] = 3

    if isinstance(data_split_savepath, Path):
        if not data_split_savepath.exists():
            if not data_split_savepath.parent.is_dir():
                data_split_savepath.parent.mkdir(parents=True)
            index_df.to_hdf(data_split_savepath, key='/data', mode='w')
        elif logger is not None:
            logger.warning(f'Indices for data split save file, {data_split_savepath}, already exists! Aborting save...')

    feature_train = train_data.loc[:, feature_vars].to_numpy()
    feature_val = val_data.loc[:, feature_vars].to_numpy()
    feature_test = test_data.loc[:, feature_vars].to_numpy()
    if scale_features:
        if not train_data.loc[:, feature_vars].empty:
            feature_train = feature_scaler.transform(train_data.loc[:, feature_vars])
        if not val_data.loc[:, feature_vars].empty:
            feature_val = feature_scaler.transform(val_data.loc[:, feature_vars])
        if not test_data.loc[:, feature_vars].empty:
            feature_test = feature_scaler.transform(test_data.loc[:, feature_vars])

    target_train = train_data.loc[:, target_vars].to_numpy()
    target_val = val_data.loc[:, target_vars].to_numpy()
    target_test = test_data.loc[:, target_vars].to_numpy()
    if scale_targets:
        if not train_data.loc[:, target_vars].empty:
            target_train = target_scaler.transform(train_data.loc[:, target_vars])
        if not val_data.loc[:, target_vars].empty:
            target_val = target_scaler.transform(val_data.loc[:, target_vars])
        if not test_data.loc[:, target_vars].empty:
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

