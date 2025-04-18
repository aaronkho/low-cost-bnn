import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import ModuleDict, Linear, BatchNorm1d, Identity, LeakyReLU, GELU
from ..utils.helpers import identity_fn
from ..utils.helpers_pytorch import default_dtype, default_device



# ------ MODELS ------


class TrainableUncertaintyAwareRegressorNN(torch.nn.Module):


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
        batch_norm=False,
        name='regressor_bnn',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self._n_units_per_channel = 1
        self._parameterization_class = param_class
        self._n_channel_outputs = 1
        if hasattr(self._parameterization_class, '_n_params'):
            self._n_channel_outputs = self._parameterization_class._n_params * self._n_units_per_channel
        self._n_recast_channel_outputs = 1
        if hasattr(self._parameterization_class, '_n_recast_params'):
            self._n_recast_channel_outputs = self._parameterization_class._n_recast_params * self._n_units_per_channel

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_commons = n_common
        self.common_nodes = [self._default_width] * self.n_commons if self.n_commons > 0 else []
        self.special_nodes = [None] * self.n_outputs
        for jj in range(len(self.special_nodes)):
            self.special_nodes[jj] = []
        self._common_l1_reg = regpar_l1 if isinstance(regpar_l1, (float, int)) else 0.0
        self._common_l2_reg = regpar_l2 if isinstance(regpar_l2, (float, int)) else 0.0
        self.rel_reg = relative_regpar if isinstance(relative_regpar, (float, int)) else 1.0
        self._special_l1_reg = self._common_l1_reg * self.rel_reg
        self._special_l2_reg = self._common_l2_reg * self.rel_reg
        self.batch_norm = True if batch_norm else False

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

        self.build()


    def build(self):

        #self._base_activation = LeakyReLU(negative_slope=0.2)
        self._base_activation = GELU()

        self._common_layers = ModuleDict()
        for ii in range(len(self.common_nodes)):
            n_prev_layer = self.n_inputs if ii == 0 else self.common_nodes[ii - 1]
            if self.batch_norm:
                self._common_layers.update({f'generalized_normalization{ii}': BatchNorm1d(n_prev_layer, eps=0.001, momentum=0.1, **self.factory_kwargs)})
            self._common_layers.update({f'generalized_layer{ii}': Linear(n_prev_layer, self.common_nodes[ii], **self.factory_kwargs)})
        if len(self._common_layers) == 0:
            self._common_layers.update({f'generalized_layer0': Identity(**self.factory_kwargs)})

        self._output_channels = ModuleDict()
        n_orig_layer = self.common_nodes[-1] if len(self.common_nodes) > 0 else self.n_inputs
        for jj in range(len(self.special_nodes)):
            channel = ModuleDict()
            for kk in range(len(self.special_nodes[jj])):
                n_prev_layer = n_orig_layer if kk == 0 else self.special_nodes[jj][kk - 1]
                if self.batch_norm:
                    channel.update({f'specialized{jj}_normalization{kk}': BatchNorm1d(n_prev_layer, eps=0.001, momentum=0.1, **self.factory_kwargs)})
                channel.update({f'specialized{jj}_layer{kk}': Linear(n_prev_layer, self.special_nodes[jj][kk], **self.factory_kwargs)})
            n_prev_layer = self.special_nodes[jj][-1] if len(self.special_nodes[jj]) > 0 else n_orig_layer
            if self.batch_norm:
                channel.update({f'parameterized{jj}_normalization0': BatchNorm1d(n_prev_layer, eps=0.001, momentum=0.1, **self.factory_kwargs)})
            channel.update({f'parameterized{jj}_layer0': self._parameterization_class(n_prev_layer, self._n_units_per_channel, **self.factory_kwargs)})
            self._output_channels.update({f'output{jj}': channel})


    def to(self, *args, **kwargs):
        other = super().to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in other.factory_kwargs:
            other.factory_kwargs['dtype'] = dtype
        if 'device' in other.factory_kwargs:
            other.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        #for ii, (name, module) in enumerate(other._common_layers.named_children()):
        #    other._common_layers[name] = other._common_layers[name].to(*args, **kwargs)
        for jj, (name, module) in enumerate(other._output_channels.named_children()):
            other._output_channels[name][f'parameterized{jj}_layer0'] = other._output_channels[name][f'parameterized{jj}_layer0'].to(*args, **kwargs)
        return other


    # Output: Shape(batch_size, n_moments, n_outputs)
    def forward(self, inputs):
        commons = inputs
        for ii in range(len(self.common_nodes)):
            if f'generalized_normalization{ii}' in self._common_layers:
                commons = self._common_layers[f'generalized_normalization{ii}'](commons)
            commons = self._common_layers[f'generalized_layer{ii}'](commons)
            commons = self._base_activation(commons)
        output_channels = []
        for jj in range(len(self.special_nodes)):
            specials = commons
            for kk in range(len(self.special_nodes[jj])):
                if f'specialized{jj}_normalization{kk}' in self._output_channels[f'output{jj}']:
                    specials = self._output_channels[f'output{jj}'][f'specialized{jj}_normalization{kk}'](specials)
                specials = self._output_channels[f'output{jj}'][f'specialized{jj}_layer{kk}'](specials)
                specials = self._base_activation(specials)
            specials = self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0'](specials)
            output_channels.append(specials)
        outputs = torch.stack(output_channels, dim=-1)
        return outputs


    # Output: Shape(batch_size, n_recast_channel_outputs, n_outputs)
    def _recast(self, outputs):
        recasts = []
        for jj, output in enumerate(torch.unbind(outputs, axis=-1)):
            recast_fn = identity_fn
            if (
                hasattr(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0'], '_recast') and
                callable(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0']._recast)
            ):
                recast_fn = self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0']._recast
            recasts.append(recast_fn(output))
        return torch.stack(recasts, axis=-1)


    @property
    def _recast_map(self):
        recast_maps = []
        for jj in range(self.n_outputs):
            recast_map = {}
            if hasattr(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0'], '_recast_map'):
                recast_map.update(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0']._recast_map)
            recast_maps.append(recast_map)
        return recast_maps


    def get_divergence_losses(self):
        losses = []
        for jj in range(self.n_outputs):
            if (
                hasattr(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer'], 'get_divergence_losses') and
                callable(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer'].get_divergence_losses)
            ):
                losses.append(self._output_channels[f'output{jj}'][f'parameterized{jj}_layer0'].get_divergence_losses())
        losses = torch.stack(losses, dim=-1)
        return losses


    def _compute_layer_regularization_losses(self):
        cl1 = torch.tensor(self._common_l1_reg, **self.factory_kwargs)
        cl2 = torch.tensor(self._common_l2_reg, **self.factory_kwargs)
        sl1 = torch.tensor(self._special_l1_reg, **self.factory_kwargs)
        sl2 = torch.tensor(self._special_l2_reg, **self.factory_kwargs)
        layer_losses = []
        for key, layer in self._common_layers.items():
            if 'layer' in key:
                layer_weights = torch.tensor(0.0, **self.factory_kwargs)
                for name, param in layer.named_parameters():
                    if 'bias' not in name:
                        layer_weights += cl1 * torch.linalg.vector_norm(param, ord=1)
                        layer_weights += cl2 * torch.linalg.vector_norm(param, ord=2)
                layer_losses.append(torch.sum(layer_weights))
        for out_key, channel in self._output_channels.items():
            for key, layer in channel.items():
                if 'layer' in key and 'parameterized' not in key:
                    layer_weights = torch.tensor(0.0, **self.factory_kwargs)
                    for name, param in layer.named_parameters():
                        if 'bias' not in name:
                            layer_weights += sl1 * torch.linalg.vector_norm(param, ord=1)
                            layer_weights += sl2 * torch.linalg.vector_norm(param, ord=2)
                    layer_losses.append(torch.sum(layer_weights))
        return torch.sum(torch.stack(layer_losses, dim=-1))


    def get_metrics_result(self):
        metrics = {}
        metrics['regularization_loss'] = self._compute_layer_regularization_losses()
        return metrics


    def get_config(self):
        param_class_config = self._parameterization_class.__name__
        config = {
            'class_name': self.__class__.__name__,
            'param_class': param_class_config,
            'n_input': self.n_inputs,
            'n_output': self.n_outputs,
            'n_common': self.n_commons,
            'common_nodes': self.common_nodes,
            'special_nodes': self.special_nodes,
            'regpar_l1': self._common_l1_reg,
            'regpar_l2': self._common_l2_reg,
            'relative_regpar': self.rel_reg,
            'batch_norm': self.batch_norm,
        }
        base_config = {key: val for key, val in self.factory_kwargs.items() if key != 'device'}
        return {**config, **base_config}


    @classmethod
    def from_config(cls, config):
        if 'class_name' in config:
            _ = config.pop('class_name')
        if 'device' in config:
            _ = config.pop('device')
        param_class_config = config.pop('param_class')
        param_class = Linear
        if param_class_config == 'DenseReparameterizationNormalInverseNormal':
            from .noise_contrastive_pytorch import DenseReparameterizationNormalInverseNormal
            param_class = DenseReparameterizationNormalInverseNormal
        elif param_class_config == 'DenseReparameterizationNormalInverseGamma':
            from .evidential_pytorch import DenseReparameterizationNormalInverseGamma
            param_class = DenseReparameterizationNormalInverseGamma
        return cls(param_class=param_class, **config)


class TrainedUncertaintyAwareRegressorNN(torch.nn.Module):


    def __init__(
        self,
        trained_model,
        input_mean,
        input_var,
        output_mean,
        output_var,
        input_tags=None,
        output_tags=None,
        name='wrapped_regressor_bnn',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._trained_model = copy.deepcopy(trained_model)
        self._trained_model.load_state_dict(trained_model.state_dict())
        self._trained_model.to(torch.device(device))
        self._trained_model.eval()

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
        self._optimizer = None

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

        self.build()


    def build(self):

        extended_output_mean = []
        for ii in range(self.n_outputs):
            temp = [self._output_mean[ii]]
            if ii < len(self._recast_map):
                while len(temp) < len(self._recast_map[ii]):
                    temp.append(0.0)
            extended_output_mean.extend(temp)
        extended_output_variance = []
        for ii in range(self.n_outputs):
            temp = [self._output_variance[ii]]
            if ii < len(self._recast_map):
                while len(temp) < len(self._recast_map[ii]):
                    temp.append(self._output_variance[ii])
            extended_output_variance.extend(temp)
        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags) and ii < len(self._recast_map):
                temp = [self._output_tags[ii] + suffix for suffix in self._recast_map[ii]]
                self._extended_output_tags.extend(temp)

        self._input_mean_tensor = torch.tensor(np.atleast_2d(self._input_mean), **self.factory_kwargs)
        self._input_var_tensor = torch.tensor(np.atleast_2d(self._input_variance), **self.factory_kwargs)
        self._output_mean_tensor = torch.tensor(np.atleast_2d(extended_output_mean), **self.factory_kwargs)
        self._output_var_tensor = torch.tensor(np.atleast_2d(extended_output_variance), **self.factory_kwargs)


    @property
    def model(self):
        return self._trained_model


    @property
    def optimizer(self):
        return self._optimizer


    @optimizer.setter
    def optimizer(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            self._optimizer = optimizer


    def to(self, *args, **kwargs):
        other = super().to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in other.factory_kwargs:
            other.factory_kwargs['dtype'] = dtype
        if 'device' in other.factory_kwargs:
            other.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        if isinstance(other._trained_model, torch.nn.Module):
            other._trained_model = other._trained_model.to(*args, **kwargs)
        if hasattr(other, '_input_mean_tensor') and isinstance(other._input_mean_tensor, torch.Tensor):
            other._input_mean_tensor = other._input_mean_tensor.to(*args, **kwargs)
        if hasattr(other, '_input_var_tensor') and isinstance(other._input_var_tensor, torch.Tensor):
            other._input_var_tensor = other._input_var_tensor.to(*args, **kwargs)
        if hasattr(other, '_output_mean_tensor') and isinstance(other._output_mean_tensor, torch.Tensor):
            other._output_mean_tensor = other._output_mean_tensor.to(*args, **kwargs)
        if hasattr(other, '_output_var_tensor') and isinstance(other._output_var_tensor, torch.Tensor):
            other._output_var_tensor = other._output_var_tensor.to(*args, **kwargs)
        return other


    # Output: Shape(batch_size, n_channel_outputs * n_outputs)
    def forward(self, inputs):
        n_recast_outputs = len(self._extended_output_tags)
        norm_inputs = (inputs - self._input_mean_tensor) / torch.sqrt(self._input_var_tensor)
        norm_outputs = self._trained_model(norm_inputs)
        recast_outputs = self._recast_fn(norm_outputs)
        shaped_outputs = torch.reshape(recast_outputs, shape=(-1, n_recast_outputs))
        outputs = (shaped_outputs * torch.sqrt(self._output_var_tensor)) + self._output_mean_tensor
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = torch.tensor(input_df.loc[:, self._input_tags].to_numpy(), **self.factory_kwargs)
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs.detach().numpy().astype(input_df.iloc[:, 0].dtype), columns=self._extended_output_tags, index=input_df.index)
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_extra')]
        return output_df.drop(drop_tags, axis=1)


    def get_divergence_losses(self):
        return self._trained_model.get_divergence_losses()


    def get_config(self):
        trained_model_config = self._trained_model.get_config()
        config = {
            'class_name': self.__class__.__name__,
            'trained_model': trained_model_config,
            'input_mean': self._input_mean,
            'input_var': self._input_variance,
            'output_mean': self._output_mean,
            'output_var': self._output_variance,
            'input_tags': self._input_tags,
            'output_tags': self._output_tags,
        }
        base_config = {key: val for key, val in self.factory_kwargs.items() if key != 'device'}
        return {**config, **base_config}


    @classmethod
    def from_config(cls, config):
        if 'class_name' in config:
            _ = config.pop('class_name')
        if 'device' in config:
            _ = config.pop('device')
        trained_model_config = config.pop('trained_model')
        trained_model = TrainableUncertaintyAwareRegressorNN.from_config(trained_model_config)
        return cls(trained_model=trained_model, **config)



class TrainableUncertaintyAwareClassifierNN(torch.nn.Module):


    def __init__(
        self,
        name='classifier_bnn',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}



class TrainedUncertaintyAwareClassifierNN(torch.nn.Module):


    def __init__(
        self,
        trained_model,
        input_mean,
        input_var,
        input_tags=None,
        output_tags=None,
        name='wrapped_classifier_bnn',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._trained_model = trained_model
        self._trained_model.load_state_dict(trained_model.state_dict())
        self._trained_model.to(torch.device(device))
        self._trained_model.eval()

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
        self._optimizer = None

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

        self.build()


    def build(self):

        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags) and ii < len(self._recast_map):
                temp = [self._output_tags[ii] + suffix for suffix in self._recast_map[ii]]
                self._extended_output_tags.extend(temp)

        self._input_mean_tensor = torch.tensor(np.atleast_2d(self._input_mean), **self.factory_kwargs)
        self._input_var_tensor = torch.tensor(np.atleast_2d(self._input_variance), **self.factory_kwargs)
        self._output_mean_tensor = torch.tensor(np.atleast_2d(extended_output_mean), **self.factory_kwargs)
        self._output_var_tensor = torch.tensor(np.atleast_2d(extended_output_variance), **self.factory_kwargs)


    @property
    def model(self):
        return self._trained_model


    @property
    def optimizer(self):
        return self._optimizer


    @optimizer.setter
    def optimizer(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            self._optimizer = optimizer


    def to(self, *args, **kwargs):
        other = super().to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in other.factory_kwargs:
            other.factory_kwargs['dtype'] = dtype
        if 'device' in other.factory_kwargs:
            other.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        if isinstance(other._trained_model, torch.nn.Module):
            other._trained_model = other._trained_model.to(*args, **kwargs)
        if hasattr(other, '_input_mean_tensor') and isinstance(other._input_mean_tensor, torch.Tensor):
            other._input_mean_tensor = other._input_mean_tensor.to(*args, **kwargs)
        if hasattr(other, '_input_var_tensor') and isinstance(other._input_var_tensor, torch.Tensor):
            other._input_var_tensor = other._input_var_tensor.to(*args, **kwargs)
        if hasattr(other, '_output_mean_tensor') and isinstance(other._output_mean_tensor, torch.Tensor):
            other._output_mean_tensor = other._output_mean_tensor.to(*args, **kwargs)
        if hasattr(other, '_output_var_tensor') and isinstance(other._output_var_tensor, torch.Tensor):
            other._output_var_tensor = other._output_var_tensor.to(*args, **kwargs)
        return other


    # Output: Shape(batch_size, n_channel_outputs * n_outputs)
    def forward(self, inputs):
        n_recast_outputs = len(self._extended_output_tags)
        norm_inputs = (inputs - self._input_mean_tensor) / torch.sqrt(self._input_var_tensor)
        norm_outputs = self._trained_model(norm_inputs)
        recast_outputs = self._recast_fn(norm_outputs)
        outputs = torch.reshape(recast_outputs, shape=(-1, n_recast_outputs))
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = torch.tensor(input_df.loc[:, self._input_tags].to_numpy(), **self.factory_kwargs)
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs.detach().numpy().astype(input_df.iloc[:, 0].dtype), columns=self._extended_output_tags, index=input_df.index)
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_extra')]
        return output_df.drop(drop_tags, axis=1)


    def get_config(self):
        trained_model_config = self._trained_model.get_config()
        config = {
            'class_name': self.__class__.__name__,
            'trained_model': trained_model_config,
            'input_mean': self._input_mean,
            'input_var': self._input_variance,
            'input_tags': self._input_tags,
            'output_tags': self._output_tags,
        }
        base_config = {key: val for key, val in self.factory_kwargs.items() if key != 'device'}
        return {**config, **base_config}


    @classmethod
    def from_config(cls, config):
        if 'class_name' in config:
            _ = config.pop('class_name')
        if 'device' in config:
            _ = config.pop('device')
        trained_model_config = config.pop('trained_model')
        trained_model = TrainableUncertaintyAwareClassifierNN.from_config(trained_model_config)
        return cls(trained_model=trained_model, **config)


