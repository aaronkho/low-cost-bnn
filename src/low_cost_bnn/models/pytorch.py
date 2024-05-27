import math
import torch
from torch.nn import Parameter, ModuleDict, Sequential, Linear, LeakyReLU, Softplus
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



class EpistemicLayer(torch.nn.Module):


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
        dtype=None
    ):

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super(EpistemicLayer, self).__init__()
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



class DenseReparameterizationNormalInverseNormal(tf.keras.layers.Layer):


    def __init__(self, units, **kwargs):
        super(DenseReparameterizationNormalInverseNormal, self).__init__(**kwargs)
        self.aleatoric_activation = Softplus()
        self.epistemic = DenseReparameterizationEpistemic(units, name=self.name+'_epistemic')
        self.aleatoric = Dense(units, activation=self.aleatoric_activation, name=self.name+'_aleatoric')


    def call(self, inputs):
        epistemic_means, epistemic_stddevs, aleatoric_samples = self.epistemic(inputs)



class LowCostBNN(torch.nn.Module):


    def __init__(self, n_input, n_output, n_hidden, n_special=None):

        super(LowCostBNN, self).__init__()

        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_hiddens = n_hidden
        self.n_specials = [self.n_hiddens[0]] * self.n_outputs
        if isinstance(n_special, (list, tuple)):
            for ii in range(n_outputs):
                self.n_specials[ii] = n_special[ii] if ii < len(n_special) else n_special[-1]

        self.build()


    def build(self):

        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.softplus = Softplus()

        self.common_layers = ModuleDict()
        for ii in range(len(self.n_hiddens)):
            n_prev_layer = self.n_inputs if ii == 0 else self.n_hiddens[ii - 1]
            self.common_layers.update({f'common{ii}': Linear(n_prev_layer, self.n_hiddens[ii])})

        self.special_layers = ModuleDict()
        self.epistemic_layers = ModuleDict()
        self.aleatoric_layers = ModuleDict()
        for ii in range(self.n_outputs):
            self.special_layers.update({f'specialized{ii}': Linear(self.n_hidden, self.n_special[ii])})
            self.epistemic_layers.update({f'model{ii}': EpistemicLayer(self.n_special[ii], 1, bias=True, kernel_prior=True)})
            self.aleatoric_layers.update({f'noise{ii}': Linear(self.n_special[ii], 1)})


    def forward(self, inputs):

        outputs = [None] * (2 * self.n_outputs)

        commons = self.leaky_relu(self.common_layers(inputs))

        for ii in range(self.n_outputs):

            specials = self.leaky_relu(self.special_layers[f'specialized{ii}'](commons))
            epistemic_means, epistemic_stddevs, sample_means = self.epistemic_layers[f'model{ii}'](specials)
            aleatoric_stddevs = self.softplus(self.aleatoric_layers[f'noise{ii}'](specials))

            model_dist = tnd.normal.Normal(epistemic_means, epistemic_stddevs)
            noise_dist = tnd.normal.Normal(sample_means, aleatoric_stddevs)

            outputs[2*ii] = tnd.independent.Independent(model_dist, 1)
            outputs[2*ii+1] = tnd.independent.Independent(noise_dist, 1)

        return outputs


    def get_divergence_losses(self):
        losses = torch.zeros(torch.Size([self.n_outputs, 2]))
        for ii in range(self.n_outputs):
            if isinstance(self.epistemic_layers[f'model{ii}'], EpistemicLayer):
                kernel_loss, bias_loss = self.epistemic_layers[f'model{ii}'].get_divergence_losses()
                if kernel_loss is not None:
                    losses[ii, 0] = kernel_loss
                if bias_loss is not None and bias_loss is not None:
                    losses[ii, 1] = bias_loss
        return losses



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



def create_model(n_input, n_output, n_hidden, n_special=None, verbosity=0):
    return LowCostBNN(n_input, n_output, n_hidden, n_special)



def create_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, verbosity=0):
    if n_outputs > 1:
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, reduction='sum')
    elif n_outputs == 1:
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


