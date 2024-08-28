import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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
    return np.mean(np.abs(targets - predictions), axis=0)


def mean_squared_error(targets, predictions):
    return np.mean(np.power(targets - predictions, 2.0), axis=0)


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


def adjusted_r2_score(targets, predictions, nreg=0):
    sample_size = float(targets.shape[0])
    adj_factor = (sample_size - 1.0) / (sample_size - nreg - 1.0)
    r2 = r2_score(targets, predictions, multioutput='raw_values').flatten()
    adjr2 = 1.0 - (1.0 - r2) * adj_factor
    return adjr2

