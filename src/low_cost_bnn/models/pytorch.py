import math
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter, ModuleDict, Sequential, Linear, LeakyReLU, Softplus
from torchvision.transforms import v2 as tnv
import torch.distributions as tnd



class NullDistribution():


    def __init__(self, null_value=None):
        self.null_value = null_value


    def sample(self):
        return self.null_value


    def mean(self):
        return self.null_value


    def stddev(self):
        return self.null_value



class DenseReparameterizationEpistemic(torch.nn.Module):


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

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.kernel_loc = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
        self.kernel_scale = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
        self.use_kernel_prior = kernel_prior
        self.use_bias_prior = bias_prior

        if bias:
            self.bias_loc = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
            self.bias_scale = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias_loc', None)
            self.register_parameter('bias_scale', None)

        self.kernel_divergence_fn = kernel_divergence_fn
        self.bias_divergence_fn = bias_divergence_fn

        self.reset_parameters()
        self.build()


    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.kernel_loc, a=math.sqrt(5))
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel_scale)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.kernel_scale, 0.01 * bound, bound)
        if self.bias_loc is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel_loc)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_loc, -1.0 * bound, bound)
        if self.bias_scale is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel_scale)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_scale, 0.01 * bound, bound)


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
        loss = torch.zeros(posterior.shape)
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


    def forward(self, inputs):
        kernel_posterior_tensor = self.kernel_posterior.sample()
        bias_posterior_tensor = self.bias_posterior.sample()
        samples = torch.matmul(inputs, kernel_posterior_tensor) + bias_posterior_tensor
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return means, stddevs, samples


    # Not sure if these are actually used in TensorFlow-equivalent model
    def get_divergence_losses(self):
        kernel_divergence_loss = self._apply_divergence(self.kernel_divergence_fn, self.kernel_posterior, self.kernel_prior)
        bias_divergence_loss = self._apply_divergence(self.bias_divergence_fn, self.bias_posterior, self.bias_prior)
        return torch.cat([kernel_divergence_loss, bias_divergence_loss], dim=1)



class DenseReparameterizationNormalInverseNormal(torch.nn.Module):


    _n_moments = 4


    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        kernel_prior=True,
        bias_prior=False,
        device=None,
        dtype=None,
        **kwargs
    ):

        super(DenseReparameterizationNormalInverseNormal, self).__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self._aleatoric_activation = Softplus(beta=1.0)
        self._epistemic = DenseReparameterizationEpistemic(in_features, out_features, bias=bias, kernel_prior=kernel_prior, bias_prior=bias_prior, **self.factory_kwargs)
        self._aleatoric = Linear(in_features, out_features, **self.factory_kwargs)


    # Output: Shape(batch_size, n_moments)
    def forward(self, inputs):
        epistemic_means, epistemic_stddevs, aleatoric_samples = self._epistemic(inputs)
        aleatoric_stddevs = self._aleatoric_activation(self._aleatoric(inputs))
        return torch.cat([epistemic_means, epistemic_stddevs, aleatoric_samples, aleatoric_stddevs], dim=-1)


    def get_divergence_losses(self):
        return self._epistemic.get_divergence_losses()



class TrainableLowCostBNN(torch.nn.Module):


    _parameterization_class = DenseReparameterizationNormalInverseNormal


    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_special=None,
        name='BNN-NCP',
        device=None,
        dtype=None,
        **kwargs
    ):

        super(TrainableLowCostBNN, self).__init__(**kwargs)

        self.name = name
        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_hiddens = list(n_hidden) if isinstance(n_hidden, (list, tuple)) else [20]
        self.n_specials = list(n_special) if isinstance(n_special, (list, tuple)) else [self.n_hiddens[0]]
        while len(self.n_specials) < self.n_outputs:
            self.n_specials.append(self.n_specials[-1])
        if len(self.n_specials) > self.n_outputs:
            self.n_specials = self.n_specials[:self.n_outputs]
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.build()


    def build(self):

        self._leaky_relu = LeakyReLU(negative_slope=0.2)

        self._common_layers = ModuleDict()
        for ii in range(len(self.n_hiddens)):
            n_prev_layer = self.n_inputs if ii == 0 else self.n_hiddens[ii - 1]
            self._common_layers.update({f'common{ii}': Linear(n_prev_layer, self.n_hiddens[ii], **self.factory_kwargs)})

        self._special_layers = ModuleDict()
        self._output_layers = ModuleDict()
        for jj in range(self.n_outputs):
            self._special_layers.update({f'specialized{jj}': Linear(self.n_hiddens[-1], self.n_specials[jj], **self.factory_kwargs)})
            self._output_layers.update({f'output{jj}': DenseReparameterizationNormalInverseNormal(self.n_specials[jj], 1, bias=True, kernel_prior=True, **self.factory_kwargs)})


    # Output: Shape(batch_size, n_moments, n_outputs)
    def forward(self, inputs):
        commons = inputs
        for ii in range(len(self.n_hiddens)):
            commons = self._leaky_relu(self._common_layers[f'common{ii}'](commons))
        output_channels = []
        for jj in range(self.n_outputs):
            specials = self._leaky_relu(self._special_layers[f'specialized{jj}'](commons))
            output_channels.append(self._output_layers[f'output{jj}'](specials))
        outputs = torch.stack(output_channels, dim=-1)
        return outputs


    def get_divergence_losses(self):
        losses = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_layers[f'output{ii}'], DenseReparameterizationNormalInverseNormal):
                losses.append(self._output_layers[f'output{ii}'].get_divergence_losses())
        losses = torch.stack(losses, dim=-1)
        return losses



class TrainedLowCostBNN(torch.nn.Module):


    def __init__(
        self,
        trained_model,
        input_mean,
        input_var,
        output_mean,
        output_var,
        input_tags=None,
        output_tags=None,
        name='Wrapped_BNN-NCP',
        device=None,
        dtype=None,
        **kwargs
    ):

        super(TrainedLowCostBNN, self).__init__(**kwargs)

        self.name = name
        self._trained_model = trained_model
        self._input_mean = input_mean
        self._input_variance = input_var
        self._output_mean = output_mean
        self._output_variance = output_var
        self._input_tags = input_tags
        self._output_tags = output_tags
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_inputs = len(self._input_mean)
        self.n_outputs = len(self._output_mean)

        self.build()


    def build(self):

        extended_output_mean = []
        for ii in range(self.n_outputs):
            temp = [self._output_mean[ii], 0.0, self._output_mean[ii], 0.0]
            extended_output_mean.extend(temp)
        output_mean = np.array(extended_output_mean)
        extended_output_variance = []
        for ii in range(self.n_outputs):
            temp = [self._output_variance[ii], self._output_variance[ii], self._output_variance[ii], self._output_variance[ii]]
            extended_output_variance.extend(temp)
        output_variance = np.array(extended_output_variance)
        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags):
                temp = [self._output_tags[ii]+'_mu', self._output_tags[ii]+'_epi_sigma', self._output_tags[ii]+'_mu_sample', self._output_tags[ii]+'_alea_sigma']
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


    # Output: Shape(batch_size, n_moments * n_outputs)
    def forward(self, inputs):
        n_moments = self._trained_model._parameterization_class._n_moments
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        shaped_outputs = torch.reshape(norm_outputs, shape=(-1, self.n_outputs * n_moments))
        outputs = self._output_denorm(shaped_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=self.factory_kwargs.get('dtype'))
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._expanded_output_tags, dtype=input_df.dtypes.iloc[0])
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_sample')]
        return output_df.drop(drop_tags, axis=1)


    def get_divergence_losses(self):
        return self._trained_model.get_divergence_losses()



class DistributionNLLLoss(torch.nn.modules.loss._Loss):


    def __init__(self, name='nll', reduction='sum'):

        super(DistributionNLLLoss, self).__init__(reduction=reduction)

        self.name = name


    def forward(self, target_values, distribution_moments):
        distributions = tnd.independent.Independent(tnd.normal.Normal(loc=distribution_moments[..., 0], scale=distribution_moments[..., 1]), 1)
        loss = -distributions.log_prob(target_values[..., 0])
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
        priors = tnd.independent.Independent(tnd.normal.Normal(loc=prior_moments[..., 0], scale=prior_moments[..., 1]), 1)
        posteriors = tnd.independent.Independent(tnd.normal.Normal(loc=posterior_moments[..., 0], scale=posterior_moments[..., 1]), 1)
        loss = tnd.kl.kl_divergence(priors, posteriors)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class NoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    def __init__(self, likelihood_weight=1.0, epistemic_weight=1.0, aleatoric_weight=1.0, name='ncp', reduction='sum'):

        super(NoiseContrastivePriorLoss, self).__init__(reduction=reduction)

        self.name = name
        self._likelihood_weights = likelihood_weight
        self._epistemic_weights = epistemic_weight
        self._aleatoric_weights = aleatoric_weight
        self._likelihood_loss_fn = DistributionNLLLoss(name=self.name+'_nll', reduction=self.reduction)
        self._epistemic_loss_fn = DistributionKLDivLoss(name=self.name+'_epi_kld', reduction=self.reduction)
        self._aleatoric_loss_fn = DistributionKLDivLoss(name=self.name+'_alea_kld', reduction=self.reduction)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = torch.tensor([self._likelihood_weights])
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_model_divergence_loss(self, targets, predictions):
        weight = torch.tensor([self._epistemic_weights])
        base = self._epistemic_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    def _calculate_noise_divergence_loss(self, targets, predictions):
        weight = torch.tensor([self._aleatoric_weights])
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


    def __init__(self, n_outputs, likelihood_weights, epistemic_weights, aleatoric_weights, name='multi_ncp', reduction='sum'):

        super(MultiOutputNoiseContrastivePriorLoss, self).__init__(reduction=reduction)

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


