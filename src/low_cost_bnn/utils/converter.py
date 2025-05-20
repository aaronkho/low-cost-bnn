import copy

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from ..models import tensorflow as tf_models

import torch
from ..models import pytorch as torch_models

from .helpers import numpy_default_dtype

tf.keras.backend.set_floatx('float32' if numpy_default_dtype == np.float32 else 'float64')
tf_default_dtype = tf.keras.backend.floatx()
tf_default_device = 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu'

torch.set_default_dtype(torch.float32 if numpy_default_dtype == np.float32 else torch.float64)
torch_default_dtype = torch.get_default_dtype()
torch_default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert_tensorflow_ncp_to_pytorch(tf_model):

    model = tf_model
    wrapped_model_dict = {}
    model_dict = {}
    weight_dict = {}
    bias_dict = {}

    if isinstance(model, tf_models.TrainedUncertaintyAwareRegressorNN):
        config = {k: v for k, v in model.get_config().items() if k != 'trained_model'}
        wrapped_model_dict.update(config)
        model = model.get_layer[0]

    if isinstance(model, tf_models.TrainableUncertaintyAwareRegressorNN):
        config = {k: v for k, v in model.get_config().items() if k != 'param_class'}
        variables = model.get_weight_paths()
        for var in variables:
            components = var.split('.')
            if components[0].startswith('kernel'):
                key = variables[var].name.split('/')[0]
                weight_dict[key] = variables[var].numpy()
            if components[0].startswith('bias'):
                key = variables[var].name.split('/')[0]
                bias_dict[key] = variables[var].numpy()
        model_dict.update(config)
        param_class_config = model.get_config().get('param_class', '')
        if param_class_config == 'DenseReparameterizationNormalInverseNormal':
            from ..models.noise_contrastive_pytorch import DenseReparameterizationNormalInverseNormal
            param_class = DenseReparameterizationNormalInverseNormal
        elif param_class_config == 'DenseReparameterizationNormalInverseGamma':
            from ..models.evidential_pytorch import DenseReparameterizationNormalInverseGamma
            param_class = DenseReparameterizationNormalInverseGamma

    return torch_model
