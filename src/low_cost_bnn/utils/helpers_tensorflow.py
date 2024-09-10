import os
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .helpers import numpy_default_dtype

tf.keras.backend.set_floatx('float64' if numpy_default_dtype == np.float64 else 'float32')
default_dtype = tf.keras.backend.floatx()


def set_tf_logging_level(level):
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel(logging.FATAL)
    elif level == logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel(logging.ERROR)
    elif level == logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel(logging.WARNING)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.get_logger().setLevel(logging.INFO)


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


def create_noise_contrastive_prior_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, distance_loss, verbosity=0):
    if n_outputs > 1:
        from ..models.noise_contrastive_tensorflow import MultiOutputNoiseContrastivePriorLoss
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, distance_loss, reduction='sum')
    elif n_outputs == 1:
        from ..models.noise_contrastive_tensorflow import NoiseContrastivePriorLoss
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, distance_loss, reduction='sum')
    else:
        raise ValueError('Number of outputs to NCP loss function generator must be an integer greater than zero.')


def create_evidential_loss_function(n_outputs, nll_weights, evi_weights, verbosity=0):
    if n_outputs > 1:
        from ..models.evidential_tensorflow import MultiOutputEvidentialLoss
        return MultiOutputEvidentialLoss(n_outputs, nll_weights, evi_weights, reduction='sum')
    elif n_outputs == 1:
        from ..models.evidential_tensorflow import EvidentialLoss
        return EvidentialLoss(nll_weights, evi_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to Evidential loss function generator must be an integer greater than zero.')


def create_cross_entropy_loss_function(n_classes, h_weights, verbosity=0):
    if n_classes > 1:
        from ..models.gaussian_process_tensorflow import MultiClassCrossEntropyLoss
        return MultiClassCrossEntropyLoss(h_weights, reduction='sum')
    elif n_classes == 1:
        from ..models.gaussian_process_tensorflow import CrossEntropyLoss
        return CrossEntropyLoss(h_weights, reduction='sum')
    else:
        raise ValueError('Number of classes to SNGP loss function generator must be an integer greater than zero.')


def create_regressor_model(
    n_input,
    n_output,
    n_common,
    common_nodes=None,
    special_nodes=None,
    regpar_l1=0.0,
    regpar_l2=0.0,
    relative_regpar=1.0,
    style='ncp',
    name=f'ncp',
    verbosity=0
):
    from ..models.tensorflow import TrainableUncertaintyAwareRegressorNN
    parameterization_layer = tf.keras.layers.Identity
    if style == 'ncp':
        from ..models.noise_contrastive_tensorflow import DenseReparameterizationNormalInverseNormal
        parameterization_layer = DenseReparameterizationNormalInverseNormal
    if style == 'evidential':
        from ..models.evidential_tensorflow import DenseReparameterizationNormalInverseGamma
        parameterization_layer = DenseReparameterizationNormalInverseGamma
    model = TrainableUncertaintyAwareRegressorNN(
        parameterization_layer,
        n_input,
        n_output,
        n_common,
        common_nodes=common_nodes,
        special_nodes=special_nodes,
        regpar_l1=regpar_l1,
        regpar_l2=regpar_l2,
        relative_regpar=relative_regpar,
        name=name
    )
    return model


def create_regressor_loss_function(n_output, style='ncp', verbosity=0, **kwargs):
    if style == 'ncp':
        return create_noise_contrastive_prior_loss_function(n_output, verbosity=verbosity, **kwargs)
    elif style == 'evidential':
        return create_evidential_loss_function(n_output, verbosity=verbosity, **kwargs)
    else:
        raise KeyError('Invalid loss function style passed to regressor loss function generator.')


def wrap_regressor_model(model, scaler_in, scaler_out):
    from ..models.tensorflow import TrainedUncertaintyAwareRegressorNN
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
    wrapper = TrainedUncertaintyAwareRegressorNN(
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


def create_classifier_model(
    n_input,
    n_output,
    n_common,
    common_nodes=None,
    special_nodes=None,
    spectral_norm=0.9,
    relative_norm=1.0,
    style='sngp',
    name=f'sngp',
    verbosity=0
):
    from ..models.tensorflow import TrainableUncertaintyAwareClassifierNN
    parameterization_layer = tf.keras.layers.Identity
    if style == 'sngp':
        from ..models.gaussian_process_tensorflow import DenseReparameterizationGaussianProcess
        parameterization_layer = DenseReparameterizationGaussianProcess
    model = TrainableUncertaintyAwareClassifierNN(
        parameterization_layer,
        n_input,
        n_output,
        n_common,
        common_nodes=common_nodes,
        special_nodes=special_nodes,
        spectral_norm=spectral_norm,
        relative_norm=relative_norm,
        name=name
    )
    return model


def create_classifier_loss_function(n_output, style='sngp', verbosity=0, **kwargs):
    if style == 'sngp':
        return create_cross_entropy_loss_function(n_output, verbosity=verbosity, **kwargs)
    else:
        raise KeyError('Invalid loss function style passed to classifier loss function generator.')


def wrap_classifier_model(model, scaler_in, names_out):
    from ..models.tensorflow import TrainedUncertaintyAwareClassifierNN
    try:
        input_mean = scaler_in.mean_
        input_var = scaler_in.var_
        input_tags = scaler_in.feature_names_in_.tolist()
        output_tags = names_out
    except:
        input_mean = np.array([0.0] * model.n_inputs)
        input_var = np.array([1.0] * model.n_inputs)
        input_tags = None
        output_tags = None
    wrapper = TrainedUncertaintyAwareClassifierNN(
        model,
        input_mean,
        input_var,
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

