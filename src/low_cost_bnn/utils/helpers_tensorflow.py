import numpy as np
import tensorflow as tf

def create_data_loader(features, targets, batch_size=None, buffer_size=None, seed=None):
    loader = tf.data.Dataset.from_tensor_slices((features, targets))
    if isinstance(buffer_size, int):
        loader = loader.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    if isinstance(batch_size, int):
        loader = loader.batch(batch_size)
    return loader


def create_learning_rate_scheduler(initial_lr, decay_steps, decay_rate):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )


def create_adam_optimizer(lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)

