import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Identity, Dense, LeakyReLU
from tensorflow.keras.regularizers import L1L2
import tensorflow_models as tfm
from ..utils.helpers import identity_fn
from ..utils.helpers_tensorflow import default_dtype



# ------ MODELS ------


class TrainableUncertaintyAwareRegressorNN(tf.keras.models.Model):


    _default_width = 512


    def __init__(
        self,
        param_class,
        n_input,
        n_output,
        n_common,
        common_nodes=None,
        special_nodes=None,
        regpar_l1=0.0,
        regpar_l2=0.0,
        relative_regpar=1.0,
        **kwargs
    ):

        if 'name' not in kwargs:
            kwargs['name'] = 'bnn'
        super(TrainableUncertaintyAwareRegressorNN, self).__init__(**kwargs)

        self._n_units_per_channel = 1
        self._parameterization_class = param_class
        self._n_channel_outputs = 1
        if hasattr(self._parameterization_class, '_n_params'):
            self._n_channel_outputs = self._parameterization_class._n_params * self._n_units_per_channel
        self._n_recast_channel_outputs = 1
        if hasattr(self._parameterization_class, '_n_recast_params'):
            self._n_recast_channel_outputs = self._parameterization_class._n_recast_params * self._n_units_per_channel

        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_commons = n_common
        self.common_nodes = [self._default_width] * self.n_commons if self.n_commons > 0 else []
        self.special_nodes = [[]] * self.n_outputs
        self._common_l1_reg = regpar_l1 if isinstance(regpar_l1, (float, int)) else 0.0
        self._common_l2_reg = regpar_l2 if isinstance(regpar_l2, (float, int)) else 0.0
        self.rel_reg = relative_regpar if isinstance(relative_regpar, (float, int)) else 1.0
        self._special_l1_reg = self._common_l1_reg * self.rel_reg
        self._special_l2_reg = self._common_l2_reg * self.rel_reg

        if isinstance(common_nodes, (list, tuple)) and len(common_nodes) > 0:
            for ii in range(self.n_commons):
                self.common_nodes[ii] = common_nodes[ii] if ii < len(common_nodes) else common_nodes[-1]

        if isinstance(special_nodes, (list, tuple)) and len(special_nodes) > 0:
            for jj in range(self.n_outputs):
                if jj < len(special_nodes) and isinstance(special_nodes[jj], (list, tuple)) and len(special_nodes[jj]) > 0:
                    for kk in range(len(special_nodes[jj])):
                        self.special_nodes[jj].append(special_nodes[jj][kk])
                elif jj > 0:
                    self.special_nodes[jj] = self.special_nodes[jj - 1]

        self._base_activation = LeakyReLU(alpha=0.2)

        self._common_layers = tf.keras.Sequential()
        for ii in range(len(self.common_nodes)):
            common_layer = Dense(
                self.common_nodes[ii],
                activation=self._base_activation,
                kernel_regularizer=L1L2(l1=self._common_l1_reg, l2=self._common_l2_reg),
                name=f'generalized_layer{ii}'
            )
            self._common_layers.add(common_layer)
        if len(self._common_layers.layers) == 0:
            self._common_layers.add(Identity(name=f'generalized_layer0'))

        self._output_channels = [None] * self.n_outputs
        for jj in range(self.n_outputs):
            channel = tf.keras.Sequential()
            for kk in range(len(self.special_nodes[jj])):
                special_layer = Dense(
                    self.special_nodes[jj][kk],
                    activation=self._base_activation,
                    kernel_regularizer=L1L2(l1=self._special_l1_reg, l2=self._special_l2_reg),
                    name=f'specialized{jj}_layer{kk}'
                )
                channel.add(special_layer)
            channel.add(self._parameterization_class(self._n_units_per_channel, name=f'parameterized{jj}_layer0'))
            self._output_channels[jj] = channel

        self.build((None, self.n_inputs))


    # Output: Shape(batch_size, n_channel_outputs, n_outputs)
    @tf.function
    def call(self, inputs):
        commons = self._common_layers(inputs)
        specials = []
        for jj in range(len(self._output_channels)):
            specials.append(self._output_channels[jj](commons))
        outputs = tf.stack(specials, axis=-1)
        return outputs


    # Output: Shape(batch_size, n_recast_channel_outputs, n_outputs)
    @tf.function
    def _recast(self, outputs):
        recasts = []
        for jj, output in enumerate(tf.unstack(outputs, axis=-1)):
            recast_fn = identity_fn
            if (
                hasattr(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0'), '_recast') and
                callable(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0')._recast)
            ):
                recast_fn = self._output_channels[jj].get_layer(f'parameterized{jj}_layer0')._recast
            recasts.append(recast_fn(output))
        return tf.stack(recasts, axis=-1)


    @property
    def _recast_map(self):
        recast_maps = []
        for jj in range(self.n_outputs):
            recast_map = {}
            if hasattr(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0'), '_recast_map'):
                recast_map.update(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0')._recast_map)
            recast_maps.append(recast_map)
        return recast_maps


    @tf.function
    def _compute_layer_regularization_losses(self):
        layer_losses = []
        for ii in range(len(self._common_layers.layers)):
            layer_losses.append(tf.reduce_sum(self._common_layers.get_layer(f'generalized_layer{ii}').losses))
        for jj in range(len(self._output_channels)):
            for kk in range(len(self._output_channels[jj].layers) - 1):
                layer_losses.append(tf.reduce_sum(self._output_channels[jj].get_layer(f'specialized{jj}_layer{kk}').losses))
        return tf.reduce_sum(tf.stack(layer_losses, axis=-1))


    @tf.function
    def get_metrics_result(self):
        metrics = super(TrainableUncertaintyAwareRegressorNN, self).get_metrics_result()
        metrics['regularization_loss'] = self._compute_layer_regularization_losses()
        return metrics


    def get_config(self):
        base_config = super(TrainableUncertaintyAwareRegressorNN, self).get_config()
        param_class_config = self._parameterization_class.__name__
        config = {
            'param_class': param_class_config,
            'n_input': self.n_inputs,
            'n_output': self.n_outputs,
            'n_common': self.n_commons,
            'common_nodes': self.common_nodes,
            'special_nodes': self.special_nodes,
            'regpar_l1': self._common_l1_reg,
            'regpar_l2': self._common_l2_reg,
            'relative_regpar': self.rel_reg,
        }
        return {**base_config, **config}


    @classmethod
    def from_config(cls, config):
        param_class_config = config.pop('param_class')
        param_class = Dense
        if param_class_config == 'DenseReparameterizationNormalInverseNormal':
            from .noise_contrastive_tensorflow import DenseReparameterizationNormalInverseNormal
            param_class = DenseReparameterizationNormalInverseNormal
        elif param_class_config == 'DenseReparameterizationNormalInverseGamma':
            from .evidential_tensorflow import DenseReparameterizationNormalInverseGamma
            param_class = DenseReparameterizationNormalInverseGamma
        return cls(param_class=param_class, **config)



class TrainedUncertaintyAwareRegressorNN(tf.keras.models.Model):

    
    def __init__(
        self,
        trained_model,
        input_mean,
        input_var,
        output_mean,
        output_var,
        input_tags=None,
        output_tags=None,
        **kwargs
    ):

        if 'name' not in kwargs:
            kwargs['name'] = 'wrapped_bnn'
        super(TrainedUncertaintyAwareRegressorNN, self).__init__(**kwargs)

        self._input_mean = input_mean
        self._input_variance = input_var
        self._output_mean = output_mean
        self._output_variance = output_var
        self._input_tags = input_tags
        self._output_tags = output_tags

        if isinstance(self._input_mean, np.ndarray):
            self._input_mean = self._input_mean.flatten().tolist()
        if isinstance(self._input_variance, np.ndarray):
            self._input_variance = self._input_variance.flatten().tolist()
        if isinstance(self._output_mean, np.ndarray):
            self._output_mean = self._output_mean.flatten().tolist()
        if isinstance(self._output_variance, np.ndarray):
            self._output_variance = self._output_variance.flatten().tolist()

        self.n_inputs = len(self._input_mean)
        self.n_outputs = len(self._output_mean)
        self._trained_model = trained_model

        self._recast_fn = identity_fn
        self._recast_map = []
        if hasattr(self._trained_model, '_recast') and callable(self._trained_model._recast):
            self._recast_fn = self._trained_model._recast
        if hasattr(self._trained_model, '_recast_map'):
            recast_maps = self._trained_model._recast_map
            for ii in range(len(recast_maps)):
                recast_map = [''] * len(recast_maps[ii])
                for key, val in recast_maps[ii].items():
                    recast_map[val] = '_' + key
                self._recast_map.append(recast_map)

        extended_output_mean = []
        for ii in range(self.n_outputs):
            temp = [self._output_mean[ii]]
            if ii < len(self._recast_map):
                while len(temp) < len(self._recast_map[ii]):
                    temp.append(0.0)
            extended_output_mean.extend(temp)
        output_mean = tf.constant(extended_output_mean, dtype=default_dtype)
        extended_output_variance = []
        for ii in range(self.n_outputs):
            temp = [self._output_variance[ii]]
            if ii < len(self._recast_map):
                while len(temp) < len(self._recast_map[ii]):
                    temp.append(self._output_variance[ii])
            extended_output_variance.extend(temp)
        output_variance = tf.constant(extended_output_variance, dtype=default_dtype)
        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags) and ii < len(self._recast_map):
                temp = [self._output_tags[ii] + suffix for suffix in self._recast_map[ii]]
                self._extended_output_tags.extend(temp)

        self._input_norm = tf.keras.layers.Normalization(axis=-1, mean=self._input_mean, variance=self._input_variance)
        self._output_denorm = tf.keras.layers.Normalization(axis=-1, mean=output_mean, variance=output_variance, invert=True)

        self.build((None, self.n_inputs))


    @property
    def get_model(self):
        return self._trained_model


    # Output: Shape(batch_size, n_channel_outputs * n_outputs)
    @tf.function
    def call(self, inputs):
        n_recast_outputs = len(self._extended_output_tags)
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        recast_outputs = self._recast_fn(norm_outputs)
        shaped_outputs = tf.reshape(recast_outputs, shape=[-1, n_recast_outputs])
        outputs = self._output_denorm(shaped_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=default_dtype)
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._extended_output_tags, dtype=input_df.dtypes.iloc[0])
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_extra')]
        return output_df.drop(drop_tags, axis=1)


    def get_config(self):
        base_config = super(TrainedUncertaintyAwareRegressorNN, self).get_config()
        trained_model_config = self._trained_model.get_config()
        config = {
            'trained_model': trained_model_config,
            'input_mean': self._input_mean,
            'input_var': self._input_variance,
            'output_mean': self._output_mean,
            'output_var': self._output_variance,
            'input_tags': self._input_tags,
            'output_tags': self._output_tags,
        }
        return {**base_config, **config}


    @classmethod
    def from_config(cls, config):
        trained_model_config = config.pop('trained_model')
        trained_model = TrainableUncertaintyAwareRegressorNN.from_config(trained_model_config)
        return cls(trained_model=trained_model, **config)


class TrainableUncertaintyAwareClassifierNN(tf.keras.models.Model):


    _default_width = 512


    def __init__(
        self,
        param_class,
        n_input,
        n_output,
        n_common,
        common_nodes=None,
        special_nodes=None,
        spectral_norm=1.0,
        relative_norm=1.0,
        **kwargs
    ):

        if 'name' not in kwargs:
            kwargs['name'] = 'bnn'
        super(TrainableUncertaintyAwareClassifierNN, self).__init__(**kwargs)

        self._n_units_per_channel = 2   # Representing two classes
        self._parameterization_class = param_class
        self._n_channel_outputs = self._n_units_per_channel
        if hasattr(self._parameterization_class, '_n_params'):
            self._n_channel_outputs = self._parameterization_class._n_params * self._n_units_per_channel
        self._n_recast_channel_outputs = self._n_units_per_channel
        if hasattr(self._parameterization_class, '_n_recast_params'):
            self._n_recast_channel_outputs = self._parameterization_class._n_recast_params * self._n_units_per_channel

        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_commons = n_common
        self.common_nodes = [self._default_width] * self.n_commons if self.n_commons > 0 else []
        self.special_nodes = [[]] * self.n_outputs
        self._common_norm = spectral_norm if isinstance(spectral_norm, (float, int)) else 1.0
        self.rel_norm = relative_norm if isinstance(relative_norm, (float, int)) else 1.0
        self._special_norm = self._common_norm * self.rel_norm

        if isinstance(common_nodes, (list, tuple)) and len(common_nodes) > 0:
            for ii in range(self.n_commons):
                self.common_nodes[ii] = common_nodes[ii] if ii < len(common_nodes) else common_nodes[-1]

        if isinstance(special_nodes, (list, tuple)) and len(special_nodes) > 0:
            for jj in range(self.n_outputs):
                if jj < len(special_nodes) and isinstance(special_nodes[jj], (list, tuple)) and len(special_nodes[jj]) > 0:
                    for kk in range(len(special_nodes[jj])):
                        self.special_nodes[jj].append(special_nodes[jj][kk])
                elif jj > 0:
                    self.special_nodes[jj] = self.special_nodes[jj - 1]

        self._base_activation = LeakyReLU(alpha=0.2)

        self._common_layers = tf.keras.Sequential()
        for ii in range(len(self.common_nodes)):
            common_layer = tfm.nlp.layers.SpectralNormalization(
                Dense(self.common_nodes[ii], activation=self._base_activation, name=f'generalized_layer{ii}'),
                iteration=1,
                norm_multiplier=self._common_norm,
                inhere_layer_name=True
            )
            self._common_layers.add(common_layer)
        if len(self._common_layers.layers) == 0:
            self._common_layers.add(Identity(name=f'generalized_layer0'))

        self._output_channels = [None] * self.n_outputs
        for jj in range(self.n_outputs):
            channel = tf.keras.Sequential()
            for kk in range(len(self.special_nodes[jj])):
                special_layer = SpectralNormalization(
                    Dense(self.special_nodes[jj][kk], activation=self._base_activation, name=f'specialized{jj}_layer{kk}'),
                    iteration=1,
                    norm_multiplier=self._special_norm,
                    inhere_layer_name=True
                )
                channel.add(special_layer)
            channel.add(self._parameterization_class(self._n_units_per_channel, name=f'parameterized{jj}_layer0'))
            self._output_channels[jj] = channel

        self.build((None, self.n_inputs))


    # Output: Shape(batch_size, n_channel_outputs, n_outputs)
    @tf.function
    def call(self, inputs):
        commons = self._common_layers(inputs)
        specials = []
        for jj in range(len(self._output_channels)):
            specials.append(self._output_channels[jj](commons))
        outputs = tf.stack(specials, axis=-1)
        return outputs


    # Output: Shape(batch_size, n_recast_channel_outputs, n_outputs)
    @tf.function
    def _recast(self, outputs):
        recasts = []
        for jj, output in enumerate(tf.unstack(outputs, axis=-1)):
            recast_fn = identity_fn
            if (
                hasattr(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0'), '_recast') and
                callable(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0')._recast)
            ):
                recast_fn = self._output_channels[jj].get_layer(f'parameterized{jj}_layer0')._recast
            recasts.append(recast_fn(output))
        return tf.stack(recasts, axis=-1)


    @property
    def _recast_map(self):
        recast_maps = []
        for jj in range(self.n_outputs):
            recast_map = {}
            if hasattr(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0'), '_recast_map'):
                recast_map.update(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0')._recast_map)
            recast_maps.append(recast_map)
        return recast_maps


    def pre_epoch_processing(self):
        for jj in range(len(self._output_channels)):
            if hasattr(self._output_channels[jj].get_layer(f'parameterized{jj}_layer0'), 'reset_covariance_matrix'):
                self._output_channels[jj].get_layer(f'parameterized{jj}_layer0').reset_covariance_matrix()


    @tf.function
    def get_metrics_result(self):
        metrics = super(TrainableUncertaintyAwareRegressorNN, self).get_metrics_result()
        metrics['regularization_loss'] = self._compute_layer_regularization_losses()
        return metrics


    def get_config(self):
        base_config = super(TrainableUncertaintyAwareClassifierNN, self).get_config()
        param_class_config = self._parameterization_class.__name__
        config = {
            'param_class': param_class_config,
            'n_input': self.n_inputs,
            'n_output': self.n_outputs,
            'n_common': self.n_commons,
            'common_nodes': self.common_nodes,
            'special_nodes': self.special_nodes,
            'relative_norm': self.rel_norm,
        }
        return {**base_config, **config}


    @classmethod
    def from_config(cls, config):
        param_class_config = config.pop('param_class')
        param_class = Dense
        if param_class_config == 'DenseReparameterizationGaussianProcess':
            from .gaussian_process_tensorflow import DenseReparameterizationGaussianProcess
            param_class = DenseReparameterizationGaussianProcess
        return cls(param_class=param_class, **config)



class TrainedUncertaintyAwareClassifierNN(tf.keras.models.Model):

    
    def __init__(
        self,
        trained_model,
        input_mean,
        input_var,
        input_tags=None,
        output_tags=None,
        **kwargs
    ):

        if 'name' not in kwargs:
            kwargs['name'] = 'wrapped_bnn'
        super(TrainedUncertaintyAwareClassifierNN, self).__init__(**kwargs)

        self._trained_model = trained_model
        self._input_mean = input_mean
        self._input_variance = input_var
        self._input_tags = input_tags
        self._output_tags = output_tags

        if isinstance(self._input_mean, np.ndarray):
            self._input_mean = self._input_mean.flatten().tolist()
        if isinstance(self._input_variance, np.ndarray):
            self._input_variance = self._input_variance.flatten().tolist()

        self.n_inputs = len(self._input_mean)
        self.n_outputs = self._trained_model.n_outputs

        self._recast_fn = identity_fn
        self._recast_map = []
        if hasattr(self._trained_model, '_recast') and callable(self._trained_model._recast):
            self._recast_fn = self._trained_model._recast
        if hasattr(self._trained_model, '_recast_map'):
            recast_maps = self._trained_model._recast_map
            for ii in range(len(recast_maps)):
                recast_map = [''] * len(recast_maps[ii])
                for key, val in recast_maps[ii].items():
                    recast_map[val] = '_' + key
                self._recast_map.append(recast_map)

        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags) and ii < len(self._recast_map):
                temp = [self._output_tags[ii] + suffix for suffix in self._recast_map[ii]]
                self._extended_output_tags.extend(temp)

        self._input_norm = tf.keras.layers.Normalization(axis=-1, mean=self._input_mean, variance=self._input_variance)

        self.build((None, self.n_inputs))


    @property
    def get_model(self):
        return self._trained_model


    # Output: Shape(batch_size, n_channel_outputs * n_outputs)
    @tf.function
    def call(self, inputs):
        n_recast_outputs = len(self._extended_output_tags)
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        recast_outputs = self._recast_fn(norm_outputs)
        outputs = tf.reshape(recast_outputs, shape=[-1, n_recast_outputs])
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=default_dtype)
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._extended_output_tags, dtype=input_df.dtypes.iloc[0])
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_extra')]
        return output_df.drop(drop_tags, axis=1)


    def get_config(self):
        base_config = super(TrainedUncertaintyAwareClassifierNN, self).get_config()
        trained_model_config = self._trained_model.get_config()
        config = {
            'trained_model': trained_model_config,
            'input_mean': self._input_mean,
            'input_var': self._input_variance,
            'input_tags': self._input_tags,
            'output_tags': self._output_tags,
        }
        return {**base_config, **config}


    @classmethod
    def from_config(cls, config):
        trained_model_config = config.pop('trained_model')
        trained_model = TrainableUncertaintyAwareClassifierNN.from_config(trained_model_config)
        return cls(trained_model=trained_model, **config)


