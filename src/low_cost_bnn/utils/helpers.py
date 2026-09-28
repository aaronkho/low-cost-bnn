import re
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

numpy_default_dtype = np.float64


def normalize(val, mu, std):
    return (val - mu) / std


def unnormalize(val, mu, std):
    return (val * std) + mu


def create_scaler(data, with_mean=True, with_std=True):
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    return scaler.fit(data)


def split(data, fraction, shuffle=True, seed=None):
    return train_test_split(data, test_size=fraction, shuffle=shuffle, random_state=seed)


def identity_fn(x):
    return x


def mean_absolute_error(targets, predictions):
    return np.mean(np.atleast_2d(np.abs(targets - predictions)), axis=0)


def mean_squared_error(targets, predictions):
    return np.mean(np.atleast_2d(np.power(targets - predictions, 2.0)), axis=0)


def relative_mean_squared_error(targets, predictions):
    fuzz = np.finfo(np.float64).eps
    return np.mean(np.atleast_2d(np.power(targets - predictions, 2.0) / (np.power(predictions, 2.0) + fuzz)), axis=0)


def fbeta_score(targets, predictions, ncls=1, thresholds=None, beta=1.0):
    thrs = [float(ii) / float(ncls + 1) for ii in range(1, ncls + 1)]
    if isinstance(thresholds, (list, tuple)):
        thrs = [float(val) for val in thresholds]
    fbeta = np.full((len(thrs), targets.shape[1]), np.nan)
    for ii in range(len(thrs)):
        tmask = (targets >= thrs[ii])
        pmask = (predictions >= thrs[ii])
        if ii < (len(thrs) - 1):
            tmask &= (targets < thrs[ii+1])
            pmask &= (predictions < thrs[ii+1])
        tp = float(np.count_nonzero(tmask & pmask))
        tn = float(np.count_nonzero(~tmask & ~pmask))
        fp = float(np.count_nonzero(~tmask & pmask))
        fn = float(np.count_nonzero(tmask & ~pmask))
        fbeta[ii] = (1.0 + beta ** 2.0) * tp / ((1.0 + beta ** 2.0) * tp + (beta ** 2.0) * fn + fp) if tp > 0 else 0.0
    return fbeta


def binary_confusion_counts(targets, predictions, thresholds):
    # Mirrors tf.keras TruePositives, TrueNegatives, FalsePositives, FalseNegatives metrics with explicit thresholds
    tmask = (np.atleast_1d(targets).flatten() != 0)
    scores = np.atleast_1d(predictions).flatten()
    thrs = np.atleast_1d(np.array(thresholds, dtype=float))
    pmask = (scores[np.newaxis, :] > thrs[:, np.newaxis])
    tp = np.count_nonzero(pmask & tmask, axis=-1).astype(float)
    tn = np.count_nonzero(~pmask & ~tmask, axis=-1).astype(float)
    fp = np.count_nonzero(pmask & ~tmask, axis=-1).astype(float)
    fn = np.count_nonzero(~pmask & tmask, axis=-1).astype(float)
    return tp, tn, fp, fn


def roc_auc_score(targets, predictions, num_thresholds=101):
    # Mirrors tf.keras AUC metric with ROC curve and trapezoidal interpolation over evenly spaced thresholds
    epsilon = 1.0e-7
    thrs = [-epsilon] + [float(ii + 1) / float(num_thresholds - 1) for ii in range(num_thresholds - 2)] + [1.0 + epsilon]
    tp, tn, fp, fn = binary_confusion_counts(targets, predictions, thrs)
    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=((tp + fn) > 0))
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=((fp + tn) > 0))
    return float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:]) / 2.0))


def adjusted_r2_score(targets, predictions, nreg=0):
    sample_size = float(targets.shape[0])
    adj_factor = (sample_size - 1.0) / (sample_size - nreg - 1.0)
    r2 = np.atleast_2d(r2_score(targets, predictions, multioutput='raw_values'))
    adjr2 = 1.0 - (1.0 - r2) * adj_factor
    return np.mean(adjr2, axis=0)


def flatten(datadict):
    odict = {}
    for key in datadict:
        if isinstance(datadict[f'{key}'], dict):
            udict = flatten(datadict[f'{key}'])
            for lkey in udict:
                odict[f'{key}.{lkey}'] = udict[lkey]
        else:
            odict[key] = copy.deepcopy(datadict[f'{key}'])
    return odict


def unflatten(datadict):
    odict = {}
    udict = {}
    for key in datadict:
        klist = key.split('.')
        if len(klist) > 1:
            nkey = '.'.join(klist[1:])
            if klist[0] not in udict:
                udict[klist[0]] = []
            udict[klist[0]].append(nkey)
        else:
            odict[klist[0]] = datadict[f'{key}']
    if udict:
        for key in udict:
            gdict = {}
            for lkey in udict[key]:
                gdict[lkey] = datadict[f'{key}.{lkey}']
            odict[key] = unflatten(gdict)
    else:
        odict = datadict
    return odict

