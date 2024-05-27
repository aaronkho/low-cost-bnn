import numpy as np
import tensorflow as tf


def create_data_loader(data_tuple, batch_size=None, buffer_size=None, seed=None):
    loader = tf.data.Dataset.from_tensor_slices(data_tuple)
    if isinstance(buffer_size, int):
        loader = loader.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    if isinstance(batch_size, int):
        loader = loader.batch(batch_size)
    return loader


def create_scheduled_adam_optimizer(model, learning_rate, decay_steps, decay_rate):
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
    return optimizer, scheduler

