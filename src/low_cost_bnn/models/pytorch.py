import math
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter, ModuleDict, Sequential, Linear, LeakyReLU, Softplus
import torchvisions as tnv
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

        self.kernel_posterior = tnd.independent.Independent(tnd.normal.Normal(self.kernel_loc, self.kernel_scale), 1)

        if self.use_kernel_prior:
            self.kernel_prior = tnd.independent.Independent(tnd.normal.Normal(torch.zeros(layer_shape), torch.ones(layer_shape)), 1)
        else:
            self.kernel_prior = NullDistribution(None)

        if self.bias_loc is not None and self.bias_scale is not None:
            self.bias_posterior = tnd.independent.Independent(tnd.normal.Normal(self.bias_loc, self.bias_scale), 1)
        else:
            self.bias_posterior = NullDistribution(None)

        if self.use_bias_prior:
            self.bias_prior = tnd.independent.Independent(tnd.normal.Normal(torch.zeros(self.out_features), torch.ones(self.out_features)), 1)
        else:
            self.bias_prior = NullDistribution(None)


    def _apply_divergence(self, divergence_fn, posterior, prior):
        loss = None
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
        return kernel_divergence_loss, bias_divergence_loss



class DenseReparameterizationNormalInverseNormal(torch.nn.Module):


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
        self._epistemic = DenseReparameterizationEpistemic(in_features, out_features, bias=bias, kernel_prior=kernel_prior, bias_prior=bias_prior, **factory_kwargs)
        self._aleatoric = Linear(in_features, out_features, **factory_kwargs)


    def forward(self, inputs):
        epistemic_means, epistemic_stddevs, aleatoric_samples = self._epistemic(inputs)
        aleatoric_stddevs = self._aleatoric_activation(self._aleatoric(inputs))
        return epistemic_means, epistemic_stddevs, aleatoric_samples, aleatoric_stddevs


    def get_divergence_losses(self):
        losses = torch.zeros(torch.Size([2]))
        kernel_loss, bias_loss = self._epistemic.get_divergence_losses()
        if kernel_loss is not None:
            losses[0] = kernel_loss
        if bias_loss is not None:
            losses[1] = bias_loss
        return losses



class TrainableLowCostBNN(torch.nn.Module):


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
        self.n_hiddens = n_hidden
        self.n_specials = [self.n_hiddens[0]] * self.n_outputs
        if isinstance(n_special, (list, tuple)):
            for ii in range(n_outputs):
                self.n_specials[ii] = n_special[ii] if ii < len(n_special) else n_special[-1]
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.build()


    def build(self):

        self._leaky_relu = LeakyReLU(negative_slope=0.2)

        self._common_layers = ModuleDict()
        for ii in range(len(self.n_hiddens)):
            n_prev_layer = self.n_inputs if ii == 0 else self.n_hiddens[ii - 1]
            self._common_layers.update({f'common{ii}': Linear(n_prev_layer, self.n_hiddens[ii], **factory_kwargs)})

        self._special_layers = ModuleDict()
        self._output_layers = ModuleDict()
        for jj in range(self.n_outputs):
            self._special_layers.update({f'specialized{jj}': Linear(self.n_hiddens[-1], self.n_specials[jj], **factory_kwargs)})
            self._output_layers.update({f'output{jj}': DenseReparameterizationNormalInverseNormal(self.n_specials[jj], 1, bias=True, kernel_prior=True, **factory_kwargs)})


    def forward(self, inputs):
        commons = inputs
        for ii in range(self.n_hiddens):
            commons = self._leaky_relu(self._common_layers[f'common{ii}'](commons))
        outputs = []
        for jj in range(self.n_outputs):
            specials = self._leaky_relu(self._special_layers[f'specialized{jj}'](commons))
            channel_outputs = self._output_layers[f'output{jj}'](specials)
            outputs.extend(list(channel_outputs))
        outputs = torch.cat(outputs, 1)
        return outputs


    def get_divergence_losses(self):
        losses = torch.zeros(torch.Size([self.n_outputs, 2]))
        for ii in range(self.n_outputs):
            if isinstance(self._output_layers[f'output{ii}'], DenseReparameterizationNormalInverseNormal):
                layer_loss = self._output_layers[f'output{ii}'].get_divergence_losses()
                losses[ii, :] = layer_loss
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

        expanded_output_mean = np.array([])
        for ii in range(self.n_outputs):
            temp = np.array([self._output_mean[ii], 0.0, self._output_mean[ii], 0.0])
            expanded_output_mean = np.hstack((expanded_output_mean, temp))
        expanded_output_variance = np.array([])
        for ii in range(self.n_outputs):
            temp = np.array([self._output_variance[ii], self._output_variance[ii], self._output_variance[ii], self._output_variance[ii]])
            expanded_output_variance = np.hstack((expanded_output_variance, temp))
        self._expanded_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags):
                temp = [self._output_tags[ii]+'_mu', self._output_tags[ii]+'_epi_sigma', self._output_tags[ii]+'_mu_sample', self._output_tags[ii]+'_alea_sigma']
                self._expanded_output_tags.extend(temp)
        self._drop_tags = [tag for tag in self._expanded_output_tags if tag.endswith('_sample')]

        adjusted_input_mean = self._input_mean
        adjusted_input_std = np.sqrt(self._input_variance)
        adjusted_output_mean = -1.0 * expanded_output_mean / np.sqrt(expanded_output_variance)
        adjusted_output_std = 1.0 / np.sqrt(expanded_output_variance)
        self._input_norm = tnv.transforms.v2.Normalize(mean=adjusted_input_mean, std=adjusted_input_std, inplace=False)
        self._output_denorm = tnv.transforms.v2.Normalize(mean=adjusted_output_mean, std=adjusted_output_variance, inplace=False)


    @property
    def get_model(self):
        return self._trained_model


    def forward(self, inputs):
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        outputs = self._output_denorm(norm_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=self.factory_kwargs.get('dtype'))
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._expanded_output_tags, dtype=input_df.dtypes.iloc[0]).drop(self._drop_tags, axis=1)
        return output_df


    def get_divergence_losses(self):
        return self._trained_model.get_divergence_losses()



class DistributionNLLLoss(torch.nn.modules.loss._Loss):


    def __init__(self, reduction='sum'):

        super(DistributionNLLLoss, self).__init__(reduction=reduction)


    def forward(self, inputs, targets):
        loss = -inputs.log_prob(targets)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class DistributionKLDivLoss(torch.nn.modules.loss._Loss):


    def __init__(self, reduction='sum'):

        super(DistributionKLDivLoss, self).__init__(reduction=reduction)


    def forward(self, inputs, targets):
        loss = tnd.kl.kl_divergence(targets, inputs)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class NoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    def __init__(self, likelihood_weight=1.0, epistemic_weight=1.0, aleatoric_weight=1.0, reduction='sum'):
        super(NoiseContrastivePriorLoss, self).__init__(reduction=reduction)
        self._likelihood_weights = torch.Tensor([likelihood_weight])
        self._epistemic_weights = torch.Tensor([epistemic_weight])
        self._aleatoric_weights = torch.Tensor([aleatoric_weight])
        self._likelihood_loss_fn = DistributionNLLLoss()
        self._epistemic_loss_fn = DistributionKLDivLoss()
        self._aleatoric_loss_fn = DistributionKLDivLoss()


    def _calculate_likelihood_loss(self, inputs, targets):
        base = self._likelihood_loss_fn(inputs, targets)
        loss = self._likelihood_weights * base
        return loss


    def _calculate_model_divergence_loss(self, inputs, targets):
        base = self._epistemic_loss_fn(inputs, targets)
        loss = self._epistemic_weights * base
        return loss


    def _calculate_noise_divergence_loss(self, inputs, targets):
        base = self._aleatoric_loss_fn(inputs, targets)
        loss = self._aleatoric_weights * base
        return loss


    def forward(self, prediction_dists, targets, model_posteriors, model_priors, noise_posteriors, noise_priors):
        likelihood_loss = self._calculate_likelihood_loss(prediction_dists, targets)
        epistemic_loss = self._calculate_model_divergence_loss(model_posteriors, model_priors)
        aleatoric_loss = self._calculate_noise_divergence_loss(noise_posteriors, noise_priors)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss



class MultiOutputNoiseContrastivePriorLoss(torch.nn.modules.loss._Loss):


    def __init__(self, n_outputs, likelihood_weights, epistemic_weights, aleatoric_weights, reduction='sum'):

        super(MultiOutputNoiseContrastivePriorLoss, self).__init__(reduction=reduction)

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
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
            self._loss_fns[ii] = NoiseContrastivePriorLoss(nll_w, epi_w, alea_w)


    def _calculate_likelihood_loss(self, inputs, targets):
        losses = torch.zeros(torch.Size([self.n_outputs]))
        for ii in range(self.n_outputs):
            losses[ii] = self._loss_fns[ii]._calculate_likelihood_loss(inputs[ii], targets[ii])
        return losses


    def _calculate_model_divergence_loss(self, inputs, targets):
        losses = torch.zeros(torch.Size([self.n_outputs]))
        for ii in range(self.n_outputs):
            losses[ii] = self._loss_fns[ii]._calculate_model_divergence_loss(inputs[ii], targets[ii])
        return losses


    def _calculate_noise_divergence_loss(self, inputs, targets):
        losses = torch.zeros(torch.Size([self.n_outputs]))
        for ii in range(self.n_outputs):
            losses[ii] = self._loss_fns[ii]._calculate_noise_divergence_loss(inputs[ii], targets[ii])
        return losses


    def forward(self, prediction_dists, targets, model_posteriors, model_priors, noise_posteriors, noise_priors):
        likelihood_loss = self._calculate_likelihood_loss(prediction_dists, targets)
        epistemic_loss = self._calculate_model_divergence_loss(model_posteriors, model_priors)
        aleatoric_loss = self._calculate_noise_divergence_loss(noise_posteriors, noise_priors)
        if self.reduction == 'mean':
            likelihood_loss = torch.mean(likelihood_loss)
            epistemic_loss = torch.mean(epistemic_loss)
            aleatoric_loss = torch.mean(aleatoric_loss)
        elif self.reduction == 'sum':
            likelihood_loss = torch.sum(likelihood_loss)
            epistemic_loss = torch.sum(epistemic_loss)
            aleatoric_loss = torch.sum(aleatoric_loss)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss


