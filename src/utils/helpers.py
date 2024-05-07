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

def split(data, fraction, shuffle=True, seed=42):
    return train_test_split(data, test_size=fraction, shuffle=shuffle, random_state=seed)
