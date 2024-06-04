import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU



# ------ MODELS ------


class TrainableUncertaintyAwareNN(tf.keras.models.Model):


    _default_width = 10


    def __init__(
        self,
        param_class,
        n_input,
        n_output,
        n_common,
        common_nodes=None,
        special_nodes=None,
        **kwargs
    ):

        super(TrainableUncertaintyAwareNN, self).__init__(**kwargs)

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
            self._common_layers.add(Dense(self.common_nodes[ii], activation=self._base_activation, name=f'common{ii}'))
        if len(self._common_layers.layers) == 0:
            self._common_layers.add(Identity(name=f'noncommon'))

        self._output_channels = [None] * self.n_outputs
        for jj in range(self.n_outputs):
            channel = tf.keras.Sequential()
            for kk in range(len(self.special_nodes[jj])):
                channel.add(Dense(self.special_nodes[jj][kk], activation=self._base_activation, name=f'specialized{jj}_layer{kk}'))
            channel.add(self._parameterization_class(self._n_units_per_channel, name=f'output{jj}'))
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
            recast_fn = tf.identity
            if hasattr(self._output_channels[jj].get_layer(f'output{jj}'), '_recast') and callable(self._output_channels[jj].get_layer(f'output{jj}')._recast):
                recast_fn = self._output_channels[jj].get_layer(f'output{jj}')._recast
            recasts.append(recast_fn(output))
        return tf.stack(recasts, axis=-1)


    def get_config(self):
        base_config = super(TrainableUncertaintyAwareNN, self).get_config()
        param_class_config = self._parameterization_class.__name__
        config = {
            'param_class': param_class_config,
            'n_input': self.n_inputs,
            'n_output': self.n_outputs,
            'n_common': self.n_commons,
            'common_nodes': self.common_nodes,
            'special_nodes': self.special_nodes,
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



class TrainedUncertaintyAwareNN(tf.keras.models.Model):

    
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

        super(TrainedUncertaintyAwareNN, self).__init__(**kwargs)

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

        n_channel_outputs = 1
        self._recast_fn = tf.identity
        self._suffixes = ['_pred_mu']
        if hasattr(self._trained_model, '_recast') and callable(self._trained_model._recast):
            self._recast_fn = self._trained_model._recast
            if hasattr(self._trained_model, '_n_recast_channel_outputs'):
                n_channel_outputs = self._trained_model._n_recast_channel_outputs
            elif hasattr(self._trained_model, '_n_channel_outputs'):
                n_channel_outputs = self._trained_model._n_channel_outputs
        elif hasattr(self._trained_model, '_n_channel_outputs'):
            n_channel_outputs = self._trained_model._n_channel_outputs
        if n_channel_outputs == 2:
            self._suffixes.append('_epi_sigma')
        elif n_channel_outputs == 3:
            self._suffixes.extend(['_epi_sigma', 'alea_sigma'])
        jj = 0
        while len(self._suffixes) < n_channel_outputs:
            self._suffixes.append(f'_parameter{jj}_extra')
            jj += 1

        extended_output_mean = []
        for ii in range(self.n_outputs):
            temp = [self._output_mean[ii]]
            while len(temp) < len(self._suffixes):
                temp.append(0.0)
            extended_output_mean.extend(temp)
        output_mean = tf.constant(extended_output_mean, dtype=tf.keras.backend.floatx())
        extended_output_variance = []
        for ii in range(self.n_outputs):
            temp = [self._output_variance[ii]]
            while len(temp) < len(self._suffixes):
                temp.append(self._output_variance[ii])
            extended_output_variance.extend(temp)
        output_variance = tf.constant(extended_output_variance, dtype=tf.keras.backend.floatx())
        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags):
                temp = [self._output_tags[ii]+suffix for suffix in self._suffixes]
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
        n_channel_outputs = len(self._suffixes)
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        recast_outputs = self._recast_fn(norm_outputs)
        shaped_outputs = tf.reshape(recast_outputs, shape=[-1, n_channel_outputs * self.n_outputs])
        outputs = self._output_denorm(shaped_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=tf.keras.backend.floatx())
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._extended_output_tags, dtype=input_df.dtypes.iloc[0])
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_extra')]
        return output_df.drop(drop_tags, axis=1)


    def get_config(self):
        base_config = super(TrainedUncertaintyAwareNN, self).get_config()
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
        trained_model = TrainableUncertaintyAwareNN.from_config(trained_model_config)
        return cls(trained_model=trained_model, **config)


