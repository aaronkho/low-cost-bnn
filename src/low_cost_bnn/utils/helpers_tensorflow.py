from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

default_dtype = tf.keras.backend.floatx()


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


def create_model(n_input, n_output, n_common, common_nodes=None, special_nodes=None, relative_reg=0.1, style='ncp', name=f'ncp', verbosity=0):
    from ..models.tensorflow import TrainableUncertaintyAwareNN
    parameterization_layer = tf.keras.layers.Identity
    if style == 'ncp':
        from ..models.noise_contrastive_tensorflow import DenseReparameterizationNormalInverseNormal
        parameterization_layer = DenseReparameterizationNormalInverseNormal
    if style == 'evidential':
        from ..models.evidential_tensorflow import DenseReparameterizationNormalInverseGamma
        parameterization_layer = DenseReparameterizationNormalInverseGamma
    model = TrainableUncertaintyAwareNN(
        parameterization_layer,
        n_input,
        n_output,
        n_common,
        common_nodes=common_nodes,
        special_nodes=special_nodes,
        relative_reg=relative_reg,
        name=name
    )
    return model


def create_loss_function(n_outputs, style='ncp', verbosity=0, **kwargs):
    if style == 'ncp':
        return create_noise_contrastive_prior_loss_function(n_outputs, verbosity=verbosity, **kwargs)
    elif style == 'evidential':
        return create_evidential_loss_function(n_outputs, verbosity=verbosity, **kwargs)
    else:
        raise KeyError('Invalid loss function style passed to loss function generator.')


def create_noise_contrastive_prior_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, verbosity=0):
    if n_outputs > 1:
        from ..models.noise_contrastive_tensorflow import MultiOutputNoiseContrastivePriorLoss
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, reduction='sum')
    elif n_outputs == 1:
        from ..models.noise_contrastive_tensorflow import NoiseContrastivePriorLoss
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


def create_evidential_loss_function(n_outputs, nll_weights, evi_weights, verbosity=0):
    if n_outputs > 1:
        from ..models.evidential_tensorflow import MultiOutputEvidentialLoss
        return MultiOutputEvidentialLoss(n_outputs, nll_weights, evi_weights, reduction='sum')
    elif n_outputs == 1:
        from ..models.evidential_tensorflow import EvidentialLoss
        return EvidentialLoss(nll_weights, evi_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


def wrap_model(model, scaler_in, scaler_out):
    from ..models.tensorflow import TrainedUncertaintyAwareNN
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
    wrapper = TrainedUncertaintyAwareNN(
        model,
        input_mean,
        input_var,
        output_mean,
        output_var,
        input_tags,
        output_tags,
        name=f'wrapped_{model.name}'
    )
    return wrapper


def load_model(model_path):
    model = None
    if isinstance(model_path, Path) and model_path.is_file():
        model = tf.keras.models.load_model(model_path)
    else:
        print(f'Specified path, {model_path}, is not a TensorFlow Keras model file! Aborting!')
    return model


def save_model(model, model_path):
    model.save(model_path)


def create_normal_posterior(mu, sigma, verbosity=0):
    return tfd.Normal(loc=mu, scale=sigma)


def create_student_t_posterior(gamma, nu, alpha, beta, verbosity=0):
    loc = gamma
    scale = tf.sqrt(beta * (1.0 + nu) / (nu * alpha))
    df = 2.0 * alpha
    return tfd.StudentT(df=df, loc=loc, scale=scale)

