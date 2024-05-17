import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Helper functions
def normalize(val, mu, std):
    return (val - mu) / std

def unnormalize(val, mu, std):
    return (val * std) + mu

def create_scaler(data, with_mean=True, with_std=True):
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    return scaler.fit(data)

def split(data, fraction, shuffle=True, seed=None):
    return train_test_split(data, test_size=fraction, shuffle=shuffle, random_state=seed)

def mean_absolute_error(targets, predictions):
    return np.mean(np.abs(targets - predictions), axis=0)

def mean_squared_error(targets, predictions):
    return np.mean(np.power(targets - predictions, 2.0), axis=0)
