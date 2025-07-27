import math
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear
from ..utils.helpers_pytorch import default_dtype, default_device, get_fuzz_factor



class DenseReparameterizationZeroUncertainty(torch.nn.Module):


    _map = {
        'mu': 0,
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma': 1,
    }
    _n_recast_params = len(_recast_map)


    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        kernel_prior=True,
        bias_prior=False,
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._n_outputs = self._n_params * self.out_features
        self._n_recast_outputs = self._n_recast_params * self.out_features

        self.dense = Linear(self.in_features, self.out_features, **self.factory_kwargs)


    def to(self, *args, **kwargs):
        other = super().to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in other.factory_kwargs:
            other.factory_kwargs['dtype'] = dtype
        if 'device' in other.factory_kwargs:
            other.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        return other


    # Output: Shape(batch_size, n_outputs)
    def forward(self, inputs):
        return torch.reshape(self.dense(inputs), shape=(-1, self.out_features))


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_zero(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.out_features, self._map['mu'] * self.out_features + self.out_features)])
        mean = torch.index_select(outputs, dim=-1, index=torch.tensor(indices, device=self.factory_kwargs.get('device', default_device)))
        stdev = torch.zeros_like(mean)
        return torch.cat([mean, stdev], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_zero(outputs)



# ------ LOSSES ------


class MSELoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='mse', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        
    def forward(self, targets, predictions):
        loss = torch.pow(predictions - targets, 2)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class RelativeMSELoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='rmse', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._fuzz = torch.tensor([get_fuzz_factor(self.factory_kwargs.get('dtype', default_dtype))], **self.factory_kwargs)


    def forward(self, targets, predictions):
        loss = torch.div(torch.pow(predictions - targets, 2), torch.pow(targets, 2) + self._fuzz)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class MixedLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        rmse_weight=1.0,
        rrmse_weight=1.0,
        name='mix',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs,
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._root_mean_square_weights = rmse_weight
        self._relative_root_mean_square_weights = rrmse_weight
        self._mean_square_loss_fn = MSELoss(name=self.name+'_mse', reduction=self.reduction, **self.factory_kwargs)
        self._relative_mean_square_loss_fn = RelativeMSELoss(name=self.name+'_rmse', reduction=self.reduction, **self.factory_kwargs)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_root_mean_square_loss(self, targets, predictions):
        weight = torch.tensor([self._root_mean_square_weights], **self.factory_kwargs)
        base = torch.sqrt(self._mean_square_loss_fn(targets, predictions))
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_relative_root_mean_square_loss(self, targets, predictions):
        weight = torch.tensor([self._relative_root_mean_square_weights], **self.factory_kwargs)
        base = torch.sqrt(self._relative_mean_square_loss_fn(targets, predictions))
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments, loss_terms) -> Output: Shape([batch_size])
    def forward(self, targets, predictions):
        rmse_loss = self._calculate_root_mean_square_loss(targets, predictions)
        rrmse_loss = self._calculate_relative_root_mean_square_loss(targets, predictions)
        total_loss = rmse_loss + rrmse_loss
        return total_loss
