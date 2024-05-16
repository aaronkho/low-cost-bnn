import math
import torch
from torch.nn import Parameter, Module, ModuleDict, Sequential, Linear, LeakyReLU, Softplus, init
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


class EpistemicLayer(Module):

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
        init.kaiming_normal_(self.kernel_loc, a=math.sqrt(5))
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.kernel_scale)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.kernel_scale, 0.01 * bound, bound)
        if self.bias_loc is not None:
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.kernel_loc)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_loc, -1.0 * bound, bound)
        if self.bias_scale is not None:
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.kernel_scale)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_scale, 0.01 * bound, bound)


    def build(self):

        layer_shape = (self.in_features, self.out_features)

        self.kernel_posterior = tnd.independent.Independent(tnd.normal.Normal(self.kernel_loc, self.kernel_scale), 2)

        if self.use_kernel_prior:
            self.kernel_prior = tnd.independent.Independent(tnd.normal.Normal(torch.zeros(layer_shape), torch.ones(layer_shape)), 2)
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
        kernel_mean = self.kernel_posterior.mean()
        kernel_stddev = self.kernel_posterior.stddev()
        bias_mean = self.bias_posterior.mean()
        dist_mean = torch.nn.functional.linear(inputs, kernel_mean, bias=bias_mean)
        dist_var = torch.nn.functional.linear(inputs ** 2, kernel_stddev ** 2)
        dist_stddev = torch.sqrt(dist_var)
        return dist_mean, dist_stddev


    def forward(self, inputs):
        kernel_posterior_tensor = self.kernel_posterior.sample()
        bias_posterior_tensor = self.bias_posterior.sample()
        samples = torch.nn.functional.linear(inputs, kernel_posterior_tensor, bias=bias_posterior_tensor)
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return means, stddevs, samples


    # Not sure if these are actually used in TensorFlow-equivalent model
    def get_divergence_losses(self):
        kernel_divergence_loss = self._apply_divergence(self.kernel_divergence_fn, self.kernel_posterior, self.kernel_prior)
        bias_divergence_loss = self._apply_divergence(self.bias_divergence_fn, self.bias_posterior, self.bias_prior)
        return kernel_divergence_loss, bias_divergence_loss


class LowCostBNN(Module):

    def __init__(self, n_inputs, n_outputs, n_hidden, n_specialized=None):

        super(LowCostBNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_special = [n_hidden] * self.n_outputs
        if isinstance(n_specialized, (list, tuple)):
            for ii in range(n_outputs):
                self.n_special[ii] = n_specialized[ii] if ii < len(n_specialized) else n_specialized[-1]
        self.build()


    def build(self):

        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.softplus = Softplus()

        self.common_layers = Linear(self.n_inputs, self.n_hidden)
        self.special_layers = ModuleDict()
        self.epistemic_layers = ModuleDict()
        self.aleatoric_layers = ModuleDict()
        for ii in range(self.n_outputs):
            self.special_layers.update({f'specialized{ii}': Linear(self.n_hidden, self.n_special[ii])})
            self.epistemic_layers.update({f'model{ii}': EpistemicLayer(self.n_special[ii], 1)})
            self.aleatoric_layers.update({f'noise{ii}': Linear(self.n_special[ii], 1)})


    def forward(self, inputs):

        outputs = [None] * (2 * self.n_outputs)

        commons = self.leaky_relu(self.common_layers(inputs))

        for ii in range(self.n_outputs):

            specials = self.leaky_relu(self.special_layers[f'specialized{ii}'](commons))
            epistemic_means, epistemic_stddevs, sample_means = self.epistemic_layers[f'model{ii}'](specials)
            aleatorics_stddevs = self.softplus(self.aleatoric_layers[f'noise{ii}'](specials))

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


# Should re-order arguments to allow built-in hidden layer flexibility
def create_model(n_inputs, n_hidden, n_outputs, n_specialized=None, verbosity=0):
    return LowCostBNN(n_inputs, n_outputs, n_hidden, n_specialized)
