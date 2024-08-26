import math
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter, Linear, LeakyReLU, Softplus
import torch.distributions as tnd
from ..utils.helpers_pytorch import default_dtype



class NullDistribution():


    def __init__(self, null_value=None):
        self.null_value = torch.tensor(np.array([null_value], dtype=float), dtype=default_dtype)


    def sample(self):
        return self.null_value


    def mean(self):
        return self.null_value


    def stddev(self):
        return self.null_value



# ------ LAYERS ------


class DenseReparameterizationEpistemic(torch.nn.Module):


    _map = {
        'mu': 0,
        'sigma': 1,
        'sample': 2
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma': 1
    }
    _n_recast_params = len(_recast_map)


    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        kernel_prior=True,
        bias_prior=False,
        kernel_divergence_fn=tnd.kl.kl_divergence,
        bias_divergence_fn=tnd.kl.kl_divergence,
        device=None,
        dtype=None,
        **kwargs
    ):

        super(DenseReparameterizationEpistemic, self).__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype if dtype is not None else default_dtype}
        self.in_features = in_features
        self.out_features = out_features

        self._n_outputs = self._n_params * self.out_features
        self._n_recast_outputs = self._n_recast_params * self.out_features

        self.kernel_loc = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
        self.kernel_scale = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
        self.use_kernel_prior = kernel_prior
        self.use_bias_prior = bias_prior

        if bias:
            self.bias_loc = Parameter(torch.empty((1, self.out_features), **self.factory_kwargs))
            self.bias_scale = Parameter(torch.empty((1, self.out_features), **self.factory_kwargs))
        else:
            self.register_parameter('bias_loc', None)
            self.register_parameter('bias_scale', None)

        self.kernel_divergence_fn = kernel_divergence_fn
        self.bias_divergence_fn = bias_divergence_fn

        # Required due to custom initialization of new trainable variables
        self.reset_parameters()
        self.build()


    def reset_parameters(self):
        kernel_scale_factor = 0.001
        bias_scale_factor = 0.001
        torch.nn.init.kaiming_normal_(self.kernel_loc, a=math.sqrt(5))
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel_scale)
        bound = kernel_scale_factor / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.kernel_scale, 0.001 * bound, bound)
        if self.bias_loc is not None:
            torch.nn.init.kaiming_uniform_(self.bias_loc, a=math.sqrt(5))
        if self.bias_scale is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.bias_scale)
            bound = bias_scale_factor / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_scale, 0.001 * bound, bound)


    def build(self):

        layer_shape = (self.in_features, self.out_features)

        self.kernel_posterior = tnd.independent.Independent(tnd.normal.Normal(loc=self.kernel_loc, scale=self.kernel_scale), 1)

        if self.use_kernel_prior:
            self.kernel_prior = tnd.independent.Independent(tnd.normal.Normal(loc=torch.zeros(layer_shape), scale=torch.ones(layer_shape)), 1)
        else:
            self.kernel_prior = NullDistribution(None)

        if self.bias_loc is not None and self.bias_scale is not None:
            self.bias_posterior = tnd.independent.Independent(tnd.normal.Normal(loc=self.bias_loc, scale=self.bias_scale), 1)
        else:
            self.bias_posterior = NullDistribution(None)

        if self.use_bias_prior:
            self.bias_prior = tnd.independent.Independent(tnd.normal.Normal(loc=torch.zeros(self.out_features), scale=torch.ones(self.out_features)), 1)
        else:
            self.bias_prior = NullDistribution(None)


    def _apply_divergence(self, divergence_fn, posterior, prior):
        loss = torch.zeros(posterior.mean.shape)
        if divergence_fn is not None and not isinstance(posterior, NullDistribution) and not isinstance(prior, NullDistribution):
            loss = divergence_fn(prior, posterior)
        return loss


    def _compute_mean_distribution_moments(self, inputs):
        kernel_mean = self.kernel_posterior.mean
        kernel_stddev = self.kernel_posterior.stddev
        bias_mean = self.bias_posterior.mean
        dist_mean = torch.matmul(inputs, kernel_mean) + bias_mean
        dist_var = torch.matmul(inputs ** 2, kernel_stddev ** 2)
        dist_stddev = torch.sqrt(dist_var)
        return dist_mean, dist_stddev


    # Output: Shape(batch_size, n_outputs)
    def forward(self, inputs):
        kernel_posterior_tensor = self.kernel_posterior.sample()
        bias_posterior_tensor = self.bias_posterior.sample()
        samples = torch.matmul(inputs, kernel_posterior_tensor) + bias_posterior_tensor
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return torch.cat([means, stddevs, samples], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_epistemic(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.out_features, self._map['mu'] * self.out_features + self.out_features)])
        indices.extend([ii for ii in range(self._map['sigma'] * self.out_features, self._map['sigma'] * self.out_features + self.out_features)])
        return torch.index_select(outputs, dim=-1, index=torch.tensor(indices))


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic(outputs)


    # Not sure if these are actually used in TensorFlow-equivalent model
    def get_divergence_losses(self, reduction='sum'):
        kernel_divergence_loss = self._apply_divergence(self.kernel_divergence_fn, self.kernel_posterior, self.kernel_prior)
        bias_divergence_loss = self._apply_divergence(self.bias_divergence_fn, self.bias_posterior, self.bias_prior)
        losses = torch.cat([kernel_divergence_loss, bias_divergence_loss], dim=-1)
        if reduction == 'mean':
            losses = torch.mean(losses)
        elif reduction == 'sum':
            losses = torch.sum(losses)
        return losses



class DenseReparameterizationNormalInverseNormal(torch.nn.Module):


    _map = {
        'mu': 0,
        'sigma_e': 1,
        'sample': 2,
        'sigma_a': 3
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma_epi': 1,
        'sigma_alea': 2
    }
    _n_recast_params = len(_recast_map)


    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        kernel_prior=True,
        bias_prior=False,
        epsilon=1.0e-6,
        device=None,
        dtype=None,
        **kwargs
    ):

        super(DenseReparameterizationNormalInverseNormal, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.factory_kwargs = {'device': device, 'dtype': dtype if dtype is not None else default_dtype}

        self._n_outputs = self._n_params * self.out_features
        self._n_recast_outputs = self._n_recast_params * self.out_features

        self.epsilon = epsilon if isinstance(epsilon, float) else 1.0e-6
        self._aleatoric_activation = Softplus(beta=1.0)
        self._epistemic = DenseReparameterizationEpistemic(self.in_features, self.out_features, bias=bias, kernel_prior=kernel_prior, bias_prior=bias_prior, **self.factory_kwargs)
        self._aleatoric = Linear(in_features, out_features, **self.factory_kwargs)


    # Output: Shape(batch_size, n_outputs)
    def forward(self, inputs):
        epistemic_outputs = self._epistemic(inputs)
        aleatoric_stddevs = self._aleatoric_activation(self._aleatoric(inputs)) + self.epsilon
        return torch.cat([epistemic_outputs, aleatoric_stddevs], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_epistemic_aleatoric(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.out_features, self._map['mu'] * self.out_features + self.out_features)])
        indices.extend([ii for ii in range(self._map['sigma_e'] * self.out_features, self._map['sigma_e'] * self.out_features + self.out_features)])
        indices.extend([ii for ii in range(self._map['sigma_a'] * self.out_features, self._map['sigma_a'] * self.out_features + self.out_features)])
        return torch.index_select(outputs, dim=-1, index=torch.tensor(indices))


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic_aleatoric(outputs)


    def get_divergence_losses(self):
        return self._epistemic.get_divergence_losses()



# ------ LOSSES ------


class DistributionNLLLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='nll', reduction='sum'):

        super(DistributionNLLLoss, self).__init__(reduction=reduction)

        self.name = name


    def forward(self, target_values, distribution_moments):
        targets, _ = torch.unbind(target_values, dim=-1)
        distribution_locs, distribution_scales = torch.unbind(distribution_moments, dim=-1)
        distributions = tnd.independent.Independent(tnd.normal.Normal(loc=distribution_locs, scale=distribution_scales), 1)
        loss = -distributions.log_prob(targets)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class DistributionKLDivLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='kld', reduction='sum'):

        super(DistributionKLDivLoss, self).__init__(reduction=reduction)

        self.name = name


    def forward(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = torch.unbind(prior_moments, dim=-1)
        posterior_locs, posterior_scales = torch.unbind(posterior_moments, dim=-1)
        priors = tnd.independent.Independent(tnd.normal.Normal(loc=prior_locs, scale=prior_scales), 1)
        posteriors = tnd.independent.Independent(tnd.normal.Normal(loc=posterior_locs, scale=posterior_scales), 1)
        loss = tnd.kl.kl_divergence(priors, posteriors)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class DistributionFisherRaoLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='fr', reduction='sum'):

        super().__init__(reduction=reduction)

        self.name = name


    def forward(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = torch.unbind(prior_moments, dim=-1)
        posterior_locs, posterior_scales = torch.unbind(posterior_moments, dim=-1)
        numerator = torch.pow(posterior_locs - prior_locs, 2) + 2.0 * torch.pow(posterior_scales - prior_scales, 2)
        denominator = torch.pow(posterior_locs + prior_locs, 2) + 2.0 * torch.pow(posterior_scales + prior_scales, 2) + 1.0e-9
        #loss = torch.atanh(torch.sqrt(numerator / denominator))
        loss = -1.0 * torch.log(1.0 - torch.sqrt(numerator / denominator))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class NoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        likelihood_weight=1.0,
        epistemic_weight=1.0,
        aleatoric_weight=1.0,
        name='ncp',
        reduction='sum',
        **kwargs
    ):

        super(NoiseContrastivePriorLoss, self).__init__(reduction=reduction, **kwargs)

        self.name = name
        self._likelihood_weights = likelihood_weight
        self._epistemic_weights = epistemic_weight
        self._aleatoric_weights = aleatoric_weight
        self._likelihood_loss_fn = DistributionNLLLoss(name=self.name+'_nll', reduction=self.reduction)
        #self._epistemic_loss_fn = DistributionKLDivLoss(name=self.name+'_epi_kld', reduction=self.reduction)
        #self._aleatoric_loss_fn = DistributionKLDivLoss(name=self.name+'_alea_kld', reduction=self.reduction)
        self._epistemic_loss_fn = DistributionFisherRaoLoss(name=self.name+'_epi_fr', reduction=self.reduction)
        self._aleatoric_loss_fn = DistributionFisherRaoLoss(name=self.name+'_alea_fr', reduction=self.reduction)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = torch.tensor([self._likelihood_weights], dtype=targets.dtype)
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_model_divergence_loss(self, targets, predictions):
        weight = torch.tensor([self._epistemic_weights], dtype=targets.dtype)
        base = self._epistemic_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_noise_divergence_loss(self, targets, predictions):
        weight = torch.tensor([self._aleatoric_weights], dtype=targets.dtype)
        base = self._aleatoric_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments, loss_terms) -> Output: Shape([batch_size])
    def forward(self, targets, predictions):
        target_values, model_prior_moments, noise_prior_moments = torch.unbind(targets, dim=-1)
        prediction_distribution_moments, model_posterior_moments, noise_posterior_moments = torch.unbind(predictions, dim=-1)
        likelihood_loss = self._calculate_likelihood_loss(target_values, prediction_distribution_moments)
        epistemic_loss = self._calculate_model_divergence_loss(model_prior_moments, model_posterior_moments)
        aleatoric_loss = self._calculate_noise_divergence_loss(noise_prior_moments, noise_posterior_moments)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss



class MultiOutputNoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        n_outputs,
        likelihood_weights,
        epistemic_weights,
        aleatoric_weights,
        name='multi_ncp',
        reduction='sum',
        **kwargs
    ):

        super(MultiOutputNoiseContrastivePriorLoss, self).__init__(reduction=reduction, **kwargs)

        self.name = name
        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._likelihood_weights = []
        self._epistemic_weights = []
        self._aleatoric_weights = []
        for ii in range(self.n_outputs):
            nll_w = 1.0
            epi_w = 1.0
            alea_w = 1.0
            if isinstance(likelihood_weights, (list, tuple)):
                nll_w = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
            if isinstance(epistemic_weights, (list, tuple)):
                epi_w = epistemic_weights[ii] if ii < len(epistemic_weights) else epistemic_weights[-1]
            if isinstance(aleatoric_weights, (list, tuple)):
                alea_w = aleatoric_weights[ii] if ii < len(aleatoric_weights) else aleatoric_weights[-1]
            self._loss_fns[ii] = NoiseContrastivePriorLoss(nll_w, epi_w, alea_w, name=f'{self.name}_out{ii}', reduction=self.reduction)
            self._likelihood_weights.append(nll_w)
            self._epistemic_weights.append(epi_w)
            self._aleatoric_weights.append(alea_w)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_likelihood_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_likelihood_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_model_divergence_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_model_divergence_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_noise_divergence_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_noise_divergence_loss(target_stack[ii], prediction_stack[ii]))
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


