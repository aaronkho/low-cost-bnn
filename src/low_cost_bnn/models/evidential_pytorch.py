import math
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter, Linear, LeakyReLU, Softplus
import torch.distributions as tnd
from ..utils.helpers_pytorch import default_dtype



# ------ LAYERS ------


class DenseReparameterizationNormalInverseGamma(torch.nn.Module):


    _map = {
        'gamma': 0,
        'nu': 1,
        'alpha': 2,
        'beta': 3
    }
    _n_params = len(_map)
    _n_recast_params = 3


    def __init__(
        self,
        in_features,
        out_features,
        device=None,
        dtype=None,
        **kwargs
    ):

        super(DenseReparameterizationNormalInverseGamma, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.factory_kwargs = {'device': device, 'dtype': dtype if dtype is not None else default_dtype}

        self._n_outputs = self._n_params * self.out_features
        self._n_recast_outputs = self._n_recast_params * self.out_features

        self._dense = Linear(self.in_features, self._n_outputs, **self.factory_kwargs)
        self._softplus = Softplus(beta=1.0)


    # Output: Shape(batch_size, n_outputs)
    def forward(self, inputs):
        outputs = self._dense(inputs)
        gamma, lognu, logalpha, logbeta = torch.tensor_split(outputs, len(self._map), dim=-1)
        nu = self._softplus(lognu)
        alpha = self._softplus(logalpha) + 1
        beta = self._softplus(logbeta)
        return torch.cat([gamma, nu, alpha, beta], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_epistemic_aleatoric(self, outputs):
        gamma_indices = [ii for ii in range(self._map['gamma'] * self.units, self._map['gamma'] * self.units + self.units)]
        nu_indices = [ii for ii in range(self._map['nu'] * self.units, self._map['nu'] * self.units + self.units)]
        alpha_indices = [ii for ii in range(self._map['alpha'] * self.units, self._map['alpha'] * self.units + self.units)]
        beta_indices = [ii for ii in range(self._map['beta'] * self.units, self._map['beta'] * self.units + self.units)]
        prediction = torch.index_select(output, dim=-1, index=torch.tensor(gamma_indices))
        ones = torch.ones(prediction.size(), dtype=output.type())
        aleatoric = torch.div(tf.index_select(output, dim=-1, index=torch.tensor(beta_indices)), torch.index_select(output, dim=-1, index=torch.tensor(alpha_indices)) - ones)
        epistemic = torch.div(aleatoric, torch.index_select(output, dim=-1, index=torch.tensor(nu_indices)))
        return torch.stack([prediction, epistemic, aleatoric], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic_aleatoric(outputs)



# ------ LOSSES ------


class NormalInverseGammaNLLLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='nll', reduction='sum', **kwargs):

        super(NormalInverseGammaNLLLoss, self).__init__(reduction=reduction, **kwargs)

        self.name = name


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def forward(self, target_values, distribution_moments):
        targets, _, _, _ = torch.unbind(target_values, dim=-1)
        gammas, nus, alphas, betas = torch.unbind(distribution_moments, dim=-1)
        omegas = 2.0 * betas * (1.0 + nus)
        loss = (
            0.5 * torch.log(np.pi / nus) -
            alphas * torch.log(omegas) +
            (alphas + 0.5) * torch.log(nus * (targets - gammas) ** 2 + omegas) +
            torch.lgamma(alphas) - torch.lgamma(alphas + 0.5)
        )
        if self.reduction == 'mean':
            loss = torch.mean(loss, dim=0)
        elif self.reduction == 'sum':
            loss = torch.sum(loss, dim=0)
        return loss



class EvidenceRegularizationLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='reg', reduction='sum', **kwargs):

        super(EvidenceRegularizationLoss, self).__init__(reduction=reduction, **kwargs)

        self.name = name


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def forward(self, target_values, distribution_moments):
        targets, _, _, _ = torch.unbind(target_values, dim=-1)
        gammas, nus, alphas, betas = torch.unbind(distribution_moments, dim=-1)
        loss = torch.abs(targets - gammas) * (2.0 * nus + alphas)
        if self.reduction == 'mean':
            loss = torch.mean(loss, dim=0)
        elif self.reduction == 'sum':
            loss = torch.sum(loss, dim=0)
        return loss



class EvidentialLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        likelihood_weight=1.0,
        evidential_weight=1.0,
        name='evidential',
        reduction='sum',
        **kwargs
    ):

        super(EvidentialLoss, self).__init__(reduction=reduction, **kwargs)

        self.name = name
        self._likelihood_weight = likelihood_weight
        self._evidential_weight = evidential_weight
        self._likelihood_loss_fn = NormalInverseGammaNLLLoss(name=self.name+'_nll', reduction=self.reduction)
        self._evidential_loss_fn = EvidenceRegularizationLoss(name=self.name+'_evi', reduction=self.reduction)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = torch.tensor([self._likelihood_weight])
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_evidential_loss(self, targets, predictions):
        weight = torch.tensor([self._evidential_weight])
        base = self._evidential_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def forward(self, targets, predictions):
        likelihood_target_values, evidential_target_values = torch.unbind(targets, dim=-1)
        likelihood_prediction_moments, evidential_prediction_moments = torch.unbind(predictions, dim=-1)
        likelihood_loss = self._calculate_likelihood_loss(likelihood_target_values, likelihood_prediction_moments)
        evidential_loss = self._calculate_evidential_loss(evidential_target_values, evidential_prediction_moments)
        total_loss = likelihood_loss + evidential_loss
        return total_loss



class MultiOutputEvidentialLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        n_outputs,
        likelihood_weights,
        evidential_weights,
        name='multi_evidential',
        reduction='sum',
        **kwargs
    ):

        super(MultiOutputEvidentialLoss, self).__init__(reduction=reduction, **kwargs)

        self.name = name
        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._likelihood_weights = []
        self._evidential_weights = []
        for ii in range(self.n_outputs):
            nll_w = 1.0
            reg_w = 1.0
            if isinstance(likelihood_weights, (list, tuple)):
                nll_w = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
            if isinstance(evidential_weights, (list, tuple)):
                reg_w = evidential_weights[ii] if ii < len(evidential_weights) else evidential_weights[-1]
            self._loss_fns[ii] = EvidentialLoss(nll_w, evi_w, name=f'{self.name}_out{ii}', reduction=self.reduction)
            self._likelihood_weights.append(nll_w)
            self._evidential_weights.append(evi_w)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_likelihood_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_likelihood_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_evidential_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_evidential_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, loss_terms, n_outputs) -> Output: Shape([batch_size])
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


