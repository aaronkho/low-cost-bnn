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
        return self.dense(inputs)


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


class SquareErrorLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='se', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

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



class RelativeSquareErrorLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='rse', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

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



class MixedSquareErrorLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        se_weight=1.0,
        rse_weight=1.0,
        name='mix',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs,
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._square_error_weight = se_weight
        self._relative_square_error_weight = rse_weight
        if isinstance(self._square_error_weight, (list, tuple, np.ndarray)):
            self._square_error_weight = self._square_error_weight[0]
        if isinstance(self._relative_square_error_weight, (list, tuple, np.ndarray)):
            self._relative_square_error_weight = self._relative_square_error_weight[0]
        self._square_error_loss_fn = SquareErrorLoss(name=self.name+'_se', reduction=self.reduction, **self.factory_kwargs)
        self._relative_square_error_loss_fn = RelativeSquareErrorLoss(name=self.name+'_rse', reduction=self.reduction, **self.factory_kwargs)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape(batch_size)
    def _calculate_square_error_loss(self, targets, predictions):
        weight = torch.tensor([self._square_error_weight], **self.factory_kwargs)
        base = self._square_error_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape(batch_size)
    def _calculate_relative_square_error_loss(self, targets, predictions):
        weight = torch.tensor([self._relative_square_error_weight], **self.factory_kwargs)
        base = self._relative_square_error_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments, loss_terms) -> Output: Shape(batch_size)
    def forward(self, targets, predictions):
        target_se_values, target_rse_values = torch.unbind(targets, dim=-1)
        prediction_se_values, prediction_rse_values = torch.unbind(predictions, dim=-1)
        se_loss = self._calculate_square_error_loss(target_se_values, prediction_se_values)
        rse_loss = self._calculate_relative_square_error_loss(target_rse_values, prediction_rse_values)
        total_loss = se_loss + rse_loss
        return total_loss



class MultiOutputMixedSquareErrorLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        n_outputs,
        se_weights,
        rse_weights,
        name='multi_mix',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs,
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._square_error_weights = []
        self._relative_square_error_weights = []
        for ii in range(self.n_outputs):
            se_w = 1.0
            rse_w = 1.0
            if isinstance(se_weights, (list, tuple, np.ndarray)):
                se_w = se_weights[ii] if ii < len(se_weights) else se_weights[-1]
            if isinstance(rse_weights, (list, tuple, np.ndarray)):
                rse_w = rse_weights[ii] if ii < len(rse_weights) else rse_weights[-1]
            self._loss_fns[ii] = MixedSquareErrorLoss(
                se_w,
                rse_w,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                **self.factory_kwargs
            )
            self._square_error_weights.append(se_w)
            self._relative_square_error_weights.append(rse_w)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape(batch_size, n_outputs)
    def _calculate_square_error_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_square_error_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape(batch_size, n_outputs)
    def _calculate_relative_square_error_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_relative_square_error_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, loss_terms, n_outputs) -> Output: Shape(batch_size, n_outputs)
    def forward(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii](target_stack[ii], prediction_stack[ii]))
        total_loss = torch.stack(losses, dim=-1)
        if self.reduction == 'mean':
            total_loss = torch.mean(total_loss)
        elif self.reduction == 'sum':
            total_loss = torch.sum(total_loss)
        return total_loss

