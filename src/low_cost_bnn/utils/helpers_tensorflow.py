import numpy as np
import tensorflow as tf
from ..models.tensorflow import TrainableLowCostBNN, TrainedLowCostBNN, NoiseContrastivePriorLoss, MultiOutputNoiseContrastivePriorLoss


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


def create_model(n_input, n_output, n_hidden=None, n_special=None, name=f'BNN-NCP', verbosity=0):
    return TrainableLowCostBNN(n_input, n_output, n_hidden, n_special, name=name)


def create_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, verbosity=0):
    if n_outputs > 1:
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, reduction='sum')
    elif n_outputs == 1:
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


def wrap_model(model, scaler_in, scaler_out):
    try:
        input_mean = scaler_in.mean_
        input_var = scaler_in.var_
        output_mean = scaler_out.mean_
        output_var = scaler_out.var_
        input_tags = scaler_in.feature_names_in_.tolist()
        output_tags = scaler_out.feature_names_in_.tolist()
    except:
        input_mean = np.array([0.0] * model.n_inputs)
        input_var = np.array([1.0] * model.n_inputs)
        output_mean = np.array([0.0] * model.n_outputs)
        output_var = np.array([1.0] * model.n_outputs)
        input_tags = None
        output_tags = None
    return TrainedLowCostBNN(model, input_mean, input_var, output_mean, output_var, input_tags, output_tags, name=f'Wrapped_{model.name}')

