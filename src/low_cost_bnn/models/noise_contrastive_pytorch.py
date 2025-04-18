import math
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter, Linear, LeakyReLU, Softplus
import torch.distributions as tnd
from ..utils.helpers_pytorch import default_dtype, default_device, get_fuzz_factor



class NullDistribution():


    def __init__(self, null_value=None, dtype=default_dtype, device=default_device):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.null_value = torch.tensor(np.array([null_value], dtype=float), **self.factory_kwargs)


    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in self.factory_kwargs:
            self.factory_kwargs['dtype'] = dtype
        if 'device' in self.factory_kwargs:
            self.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.null_value.to(*args, **kwargs)
        return self


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
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype}
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
        #fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel_scale)
        #bound = kernel_scale_factor / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.kaiming_uniform_(self.kernel_scale, a=kernel_scale_factor)
        if self.bias_loc is not None:
            torch.nn.init.kaiming_uniform_(self.bias_loc, a=math.sqrt(5))
        if self.bias_scale is not None:
            #fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.bias_scale)
            #bound = bias_scale_factor / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.kaiming_uniform_(self.bias_scale, a=bias_scale_factor)


    def build(self):

        layer_shape = (self.in_features, self.out_features)

        if self.use_kernel_prior:
            self.kernel_prior = tnd.independent.Independent(tnd.normal.Normal(
                loc=torch.zeros(layer_shape, **self.factory_kwargs),
                scale=torch.ones(layer_shape, **self.factory_kwargs)
            ), 1)
        else:
            self.kernel_prior = NullDistribution(None, **self.factory_kwargs)

        if self.use_bias_prior:
            self.bias_prior = tnd.independent.Independent(tnd.normal.Normal(
                loc=torch.zeros(self.out_features, **self.factory_kwargs),
                scale=torch.ones(self.out_features, **self.factory_kwargs)
            ), 1)
        else:
            self.bias_prior = NullDistribution(None, **self.factory_kwargs)


    def construct_posteriors(self, kernel_loc, kernel_scale, bias_loc=None, bias_scale=None):
        kernel_posterior = tnd.independent.Independent(tnd.normal.Normal(loc=kernel_loc, scale=kernel_scale), 1)
        if bias_loc is not None and bias_scale is not None:
            bias_posterior = tnd.independent.Independent(tnd.normal.Normal(loc=bias_loc, scale=bias_scale), 1)
        else:
            bias_posterior = NullDistribution(None, **self.factory_kwargs)
        return kernel_posterior, bias_posterior


    def _apply_divergence(self, divergence_fn, posterior, prior):
        loss = torch.zeros(posterior.mean.shape, **self.factory_kwargs)
        if callable(divergence_fn) and not isinstance(posterior, NullDistribution) and not isinstance(prior, NullDistribution):
            loss = divergence_fn(prior, posterior)
        return loss


    def _compute_mean_distribution_moments(self, inputs, kernel_posterior, bias_posterior):
        kernel_mean = kernel_posterior.mean
        kernel_stddev = kernel_posterior.stddev
        bias_mean = bias_posterior.mean
        dist_mean = torch.matmul(inputs, kernel_mean) + bias_mean
        dist_var = torch.matmul(inputs ** 2, kernel_stddev ** 2)
        dist_stddev = torch.sqrt(dist_var)
        return dist_mean, dist_stddev


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
        kernel_scale_plus = torch.nn.functional.softplus(self.kernel_scale)
        bias_scale_plus = torch.nn.functional.softplus(self.bias_scale) if self.bias_scale is not None else None
        kernel_posterior, bias_posterior = self.construct_posteriors(self.kernel_loc, kernel_scale_plus, self.bias_loc, bias_scale_plus)
        kernel_posterior_tensor = kernel_posterior.sample()
        bias_posterior_tensor = bias_posterior.sample()
        samples = torch.matmul(inputs, kernel_posterior_tensor) + bias_posterior_tensor
        means, stddevs = self._compute_mean_distribution_moments(inputs, kernel_posterior, bias_posterior)
        return torch.cat([means, stddevs, samples], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_epistemic(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.out_features, self._map['mu'] * self.out_features + self.out_features)])
        indices.extend([ii for ii in range(self._map['sigma'] * self.out_features, self._map['sigma'] * self.out_features + self.out_features)])
        return torch.index_select(outputs, dim=-1, index=torch.tensor(indices, device=self.factory_kwargs.get('device', default_device)))


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic(outputs)


    # Not sure if these are actually used in TensorFlow-equivalent model
    def get_divergence_losses(self, reduction='sum'):
        kernel_scale_plus = torch.nn.functional.softplus(self.kernel_scale)
        bias_scale_plus = torch.nn.functional.softplus(self.bias_scale) if self.bias_scale is not None else None
        kernel_posterior, bias_posterior = self.construct_posteriors(self.kernel_loc, kernel_scale_plus, self.bias_loc, bias_scale_plus)
        kernel_divergence_loss = self._apply_divergence(self.kernel_divergence_fn, kernel_posterior, self.kernel_prior)
        bias_divergence_loss = self._apply_divergence(self.bias_divergence_fn, bias_posterior, self.bias_prior)
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

        self._fuzz = torch.tensor([get_fuzz_factor(self.factory_kwargs.get('dtype', default_dtype))], **self.factory_kwargs)
        self._softplus = Softplus(beta=1.0)
        self._epistemic = DenseReparameterizationEpistemic(self.in_features, self.out_features, bias=bias, kernel_prior=kernel_prior, bias_prior=bias_prior, **self.factory_kwargs)
        self._aleatoric = Linear(in_features, out_features, **self.factory_kwargs)


    def to(self, *args, **kwargs):
        other = super().to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in other.factory_kwargs:
            other.factory_kwargs['dtype'] = dtype
        if 'device' in other.factory_kwargs:
            other.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        if hasattr(other, '_fuzz') and isinstance(other._fuzz, torch.Tensor):
            other._fuzz = other._fuzz.to(*args, **kwargs)
        if hasattr(other, '_epistemic') and isinstance(other._epistemic, torch.nn.Module):
            other._epistemic = other._epistemic.to(*args, **kwargs)
        return other


    # Output: Shape(batch_size, n_outputs)
    def forward(self, inputs):
        epistemic_outputs = self._epistemic(inputs)
        aleatoric_stddevs = self._softplus(self._aleatoric(inputs)) + self._fuzz
        return torch.cat([epistemic_outputs, aleatoric_stddevs], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_epistemic_aleatoric(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.out_features, self._map['mu'] * self.out_features + self.out_features)])
        indices.extend([ii for ii in range(self._map['sigma_e'] * self.out_features, self._map['sigma_e'] * self.out_features + self.out_features)])
        indices.extend([ii for ii in range(self._map['sigma_a'] * self.out_features, self._map['sigma_a'] * self.out_features + self.out_features)])
        return torch.index_select(outputs, dim=-1, index=torch.tensor(indices, device=self.factory_kwargs.get('device', default_device)))


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic_aleatoric(outputs)


    def get_divergence_losses(self):
        return self._epistemic.get_divergence_losses()



# ------ LOSSES ------


class NormalNLLLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='nll', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._fuzz = torch.tensor([get_fuzz_factor(self.factory_kwargs.get('dtype', default_dtype))], **self.factory_kwargs)


    def forward(self, target_values, distribution_moments):
        targets, _ = torch.unbind(target_values, dim=-1)
        distribution_locs, distribution_scales = torch.unbind(distribution_moments, dim=-1)
        log_prefactor = torch.log(2.0 * np.pi * torch.pow(distribution_scales, 2) + self._fuzz)
        log_shape = torch.div(torch.pow(targets - distribution_locs, 2), torch.pow(distribution_scales, 2) + self._fuzz)
        loss = 0.5 * (log_prefactor + log_shape)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class NormalNormalKLDivLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='kld', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}


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



class NormalNormalFisherRaoLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='fr', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}


    def forward(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = torch.unbind(prior_moments, dim=-1)
        posterior_locs, posterior_scales = torch.unbind(posterior_moments, dim=-1)
        distances = torch.pow(posterior_locs - prior_locs, 2)
        numerator_scales = torch.pow(posterior_scales - prior_scales, 2)
        denominator_scales = torch.pow(posterior_scales + prior_scales, 2)
        numerator = distances + 2.0 * numerator_scales
        denominator = distances + 2.0 * denominator_scales
        argument = torch.div(torch.sqrt(numerator), torch.sqrt(denominator))
        argument[argument != argument] = 0.0
        loss = torch.atanh(argument)
        #loss = -1.0 * torch.log(1.0 - argument)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class NormalNormalHighUncertaintyLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='unc', reduction='sum', dtype=default_dtype, device=default_device, **kwargs):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._fuzz = torch.tensor([get_fuzz_factor(self.factory_kwargs.get('dtype', default_dtype))], **self.factory_kwargs)


    def forward(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = torch.unbind(prior_moments, dim=-1)
        posterior_locs, posterior_scales = torch.unbind(posterior_moments, dim=-1)
        #loss = torch.sqrt(torch.div(torch.pow(posterior_scales, 2), torch.pow(posterior_locs, 2)))
        loss = torch.pow(torch.log(torch.div(posterior_scales + self._fuzz, prior_scales)), 2)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class NoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    _possible_distance_losses = [
        'fisher_rao',
        'kl_divergence',
    ]


    def __init__(
        self,
        likelihood_weight=1.0,
        epistemic_weight=1.0,
        aleatoric_weight=1.0,
        distance_loss='fisher_rao',
        name='ncp',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._likelihood_weights = likelihood_weight
        self._epistemic_weights = epistemic_weight
        self._aleatoric_weights = aleatoric_weight
        self._distance_loss = distance_loss if distance_loss in self._possible_distance_losses else self._possible_distance_losses[0]
        self._likelihood_loss_fn = NormalNLLLoss(name=self.name+'_nll', reduction=self.reduction, **self.factory_kwargs)
        if distance_loss == 'kl_divergence':
            self._epistemic_loss_fn = NormalNormalKLDivLoss(name=self.name+'_epi_kld', reduction=self.reduction, **self.factory_kwargs)
            self._aleatoric_loss_fn = NormalNormalKLDivLoss(name=self.name+'_alea_kld', reduction=self.reduction, **self.factory_kwargs)
        else:  # 'fisher_rao'
            self._epistemic_loss_fn = NormalNormalFisherRaoLoss(name=self.name+'_epi_fr', reduction=self.reduction, **self.factory_kwargs)
            #self._aleatoric_loss_fn = NormalNormalFisherRaoLoss(name=self.name+'_alea_fr', reduction=self.reduction, **self.factory_kwargs)
            self._aleatoric_loss_fn = NormalNormalHighUncertaintyLoss(name=self.name+'_alea_unc', reduction=self.reduction, **self.factory_kwargs)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = torch.tensor([self._likelihood_weights], **self.factory_kwargs)
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_model_distance_loss(self, targets, predictions):
        weight = torch.tensor([self._epistemic_weights], **self.factory_kwargs)
        base = self._epistemic_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_noise_distance_loss(self, targets, predictions):
        weight = torch.tensor([self._aleatoric_weights], **self.factory_kwargs)
        base = self._aleatoric_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments, loss_terms) -> Output: Shape([batch_size])
    def forward(self, targets, predictions):
        target_values, model_prior_moments, noise_prior_moments = torch.unbind(targets, dim=-1)
        prediction_distribution_moments, model_posterior_moments, noise_posterior_moments = torch.unbind(predictions, dim=-1)
        likelihood_loss = self._calculate_likelihood_loss(target_values, prediction_distribution_moments)
        epistemic_loss = self._calculate_model_distance_loss(model_prior_moments, model_posterior_moments)
        aleatoric_loss = self._calculate_noise_distance_loss(noise_prior_moments, noise_posterior_moments)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss



class MultiOutputNoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    _possible_distance_losses = [
        'fisher_rao',
        'kl_divergence',
    ]


    def __init__(
        self,
        n_outputs,
        likelihood_weights,
        epistemic_weights,
        aleatoric_weights,
        distance_loss='fisher_rao',
        name='multi_ncp',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._likelihood_weights = []
        self._epistemic_weights = []
        self._aleatoric_weights = []
        self._distance_loss = distance_loss if distance_loss in self._possible_distance_losses else self._possible_distance_losses[0]
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
            self._loss_fns[ii] = NoiseContrastivePriorLoss(
                nll_w,
                epi_w,
                alea_w,
                self._distance_loss,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                **self.factory_kwargs
            )
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
    def _calculate_model_distance_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_model_distance_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_noise_distance_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_noise_distance_loss(target_stack[ii], prediction_stack[ii]))
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


