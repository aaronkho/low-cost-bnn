import numpy as np
import pandas as pd
import torch
from torch.nn import ModuleDict, Linear, Identity, LeakyReLU
from torchvision.transforms import v2 as tnv
from ..utils.helpers import identity_fn



# ------ MODELS ------


class TrainableUncertaintyAwareNN(torch.nn.Module):


    _default_width = 10
    _common_regpar = 1.0


    def __init__(
        self,
        param_class,
        n_input,
        n_output,
        n_common,
        common_nodes=None,
        special_nodes=None,
        relative_reg=0.1,
        name='bnn',
        device=None,
        dtype=None,
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

        self.name = name
        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_commons = n_common
        self.common_nodes = [self._default_width] * self.n_commons if self.n_commons > 0 else []
        self.special_nodes = [[]] * self.n_outputs
        self.rel_reg = relative_reg if isinstance(relative_reg, (float, int)) else 0.1
        self._special_regpar = self._common_regpar * self.rel_reg

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

        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.build()


    def build(self):

        self._leaky_relu = LeakyReLU(negative_slope=0.2)

        self._common_layers = ModuleDict()
        for ii in range(len(self.common_nodes)):
            n_prev_layer = self.n_inputs if ii == 0 else self.common_nodes[ii - 1]
            self._common_layers.update({f'common{ii}': Linear(n_prev_layer, self.common_nodes[ii], **self.factory_kwargs)})
        if len(self._common_layers) == 0:
            self._common_layers.update({f'noncommon': Identity(**self.factory_kwargs)})

        self._output_channels = ModuleDict()
        n_orig_layer = self.common_nodes[-1] if len(self.common_nodes) > 0 else self.n_inputs
        for jj in range(self.n_outputs):
            channel = ModuleDict()
            for kk in range(len(self.special_nodes[jj])):
                n_prev_layer = n_orig_layer if kk == 0 else self.special_nodes[jj][kk - 1]
                channel.update({f'specialized{jj}_layer{kk}': Linear(n_prev_layer, self.special_nodes[jj][kk], **self.factory_kwargs)})
            n_prev_layer = self.special_nodes[jj][-1] if len(self.special_nodes[jj]) > 0 else n_orig_layer
            channel.update({f'output{jj}': self._parameterization_class(n_prev_layer, self._n_units_per_channel, **self.factory_kwargs)})
            self._output_channels.update({f'output_channel{jj}': channel})


    # Output: Shape(batch_size, n_moments, n_outputs)
    def forward(self, inputs):
        commons = inputs
        for ii in range(self.n_commons):
            commons = self._common_layers[f'common{ii}'](commons)
            commons = self._leaky_relu(commons)
        output_channels = []
        for jj in range(self.n_outputs):
            specials = commons
            for kk in range(len(self._output_channels[f'output_channel{jj}']) - 1):
                specials = self._output_channels[f'output_channel{jj}'][f'specialized{jj}_layer{kk}'](specials)
                specials = self._leaky_relu(specials)
            specials = self._output_channels[f'output_channel{jj}'][f'output{jj}'](specials)
            output_channels.append(specials)
        outputs = torch.stack(output_channels, dim=-1)
        return outputs


    # Output: Shape(batch_size, n_recast_channel_outputs, n_outputs)
    def _recast(self, outputs):
        recasts = []
        for jj, output in enumerate(torch.unbind(outputs, axis=-1)):
            recast_fn = identity_fn
            if hasattr(self._output_channels[f'output_channel{jj}'][f'output{jj}'], '_recast') and callable(self._output_channels[f'output_channel{jj}'][f'output{jj}']._recast):
                recast_fn = self._output_channels[f'output_channel{jj}'][f'output{jj}']._recast
            recasts.append(recast_fn(output))
        return torch.stack(recasts, axis=-1)


    def get_divergence_losses(self):
        losses = []
        for jj in range(self.n_outputs):
            if hasattr(self._output_channels[f'output_channel{jj}'][f'output{jj}'], 'get_divergence_losses') and callable(self._output_channels[f'output_channel{jj}'][f'output{jj}'].get_divergence_losses):
                losses.append(self._output_channels[f'output_channel{jj}'][f'output{jj}'].get_divergence_losses())
        losses = torch.stack(losses, dim=-1)
        return losses


    def _compute_layer_regularization_losses(self):
        layer_losses = []
        for ii in range(len(self._common_layers)):
            layer_weights = torch.tensor(0.0, dtype=self.factory_kwargs.get('dtype'))
            for param in self._common_layers[f'common{ii}'].parameters():
                layer_weights += self._common_regpar * torch.linalg.vector_norm(param, ord=1)
                layer_weights += self._common_regpar * torch.linalg.vector_norm(param, ord=2)
            layer_losses.append(torch.sum(layer_weights))
        for jj in range(len(self._output_channels)):
            for kk in range(len(self._output_channels[f'output_channel{jj}']) - 1):
                layer_weights = torch.tensor(0.0, dtype=self.factory_kwargs.get('dtype'))
                for param in self._output_channels[f'output_channel{jj}'][f'specialized{jj}_layer{kk}'].parameters():
                    layer_weights += self._special_regpar * torch.linalg.vector_norm(param, ord=1)
                    layer_weights += self._special_regpar * torch.linalg.vector_norm(param, ord=2)
                layer_losses.append(tf.reduce_sum(layer_weights))
        return torch.sum(torch.stack(layer_losses, dim=-1))


    def get_metrics_result(self):
        metrics = {}
        metrics['regularization_loss'] = self._compute_layer_regularization_losses()
        return metrics



class TrainedUncertaintyAwareNN(torch.nn.Module):


    def __init__(
        self,
        trained_model,
        input_mean,
        input_var,
        output_mean,
        output_var,
        input_tags=None,
        output_tags=None,
        name='wrapped_bnn',
        device=None,
        dtype=None,
        **kwargs
    ):

        super(TrainedUncertaintyAwareNN, self).__init__(**kwargs)

        self.name = name
        self._trained_model = trained_model
        self._input_mean = input_mean
        self._input_variance = input_var
        self._output_mean = output_mean
        self._output_variance = output_var
        self._input_tags = input_tags
        self._output_tags = output_tags
        self.factory_kwargs = {'device': device, 'dtype': dtype}

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

        n_channel_outputs = 1
        self._recast_fn = identity_fn
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
            self._suffixes.extend(['_epi_sigma', '_alea_sigma'])
        jj = 0
        while len(self._suffixes) < n_channel_outputs:
            self._suffixes.append(f'_parameter{jj}_extra')
            jj += 1

        self.build()


    def build(self):

        extended_output_mean = []
        for ii in range(self.n_outputs):
            temp = [self._output_mean[ii]]
            while len(temp) < len(self._suffixes):
                temp.append(0.0)
            extended_output_mean.extend(temp)
        output_mean = np.array(extended_output_mean)
        extended_output_variance = []
        for ii in range(self.n_outputs):
            temp = [self._output_variance[ii]]
            while len(temp) < len(self._suffixes):
                temp.append(self._output_variance[ii])
            extended_output_variance.extend(temp)
        output_variance = np.array(extended_output_variance)
        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags):
                temp = [self._output_tags[ii]+suffix for suffix in self._suffixes]
                self._extended_output_tags.extend(temp)

        adjusted_input_mean = torch.tensor(self._input_mean, dtype=self.factory_kwargs.get('dtype'))
        adjusted_input_std = torch.tensor(np.sqrt(self._input_variance), dtype=self.factory_kwargs.get('dtype'))
        adjusted_output_mean = torch.tensor(-1.0 * output_mean / np.sqrt(output_variance), dtype=self.factory_kwargs.get('dtype'))
        adjusted_output_std = torch.tensor(1.0 / np.sqrt(output_variance), dtype=self.factory_kwargs.get('dtype'))
        self._input_norm = tnv.Normalize(mean=adjusted_input_mean, std=adjusted_input_std, inplace=False)
        self._output_denorm = tnv.Normalize(mean=adjusted_output_mean, std=adjusted_output_std, inplace=False)


    @property
    def get_model(self):
        return self._trained_model


    # Output: Shape(batch_size, n_channel_outputs * n_outputs)
    def forward(self, inputs):
        n_channel_outputs = len(self._suffixes)
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        recast_outputs = self._recast_fn(norm_outputs)
        shaped_outputs = torch.reshape(recast_outputs, shape=(-1, n_channel_outputs * self.n_outputs))
        outputs = self._output_denorm(shaped_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=self.factory_kwargs.get('dtype'))
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._extended_output_tags, dtype=input_df.dtypes.iloc[0])
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_extra')]
        return output_df.drop(drop_tags, axis=1)


    def get_divergence_losses(self):
        return self._trained_model.get_divergence_losses()


