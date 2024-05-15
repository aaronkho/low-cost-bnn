import torch
from torch.nn import Parameter, Module, ModuleDict, Sequential, Linear, LeakyReLU, SoftPlus
import torch.distributions as tnd


class EpistemicLayer(Module):

    def __init__(
        self,
        in_features,
        out_features,
        kernel_prior_fn,
        kernel_posterior_fn=tnd.multivariate_normal.MultivariateNormal,
        kernel_divergence_fn=torch.nn.functional.kl_div,
        bias_prior_fn,
        bias_posterior_fn,
        bias_divergence_fn=torch.nn.functional.kl_div,
        device=None,
        dtype=None
    ):

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super(EpistemicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_prior_fn = kernel_prior_fn
        self.kernel_posterior_fn = kernel_posterior_fn
        self.kernel_divergence_fn = kernel_divergence_fn
        self.bias_prior_fn = bias_prior_fn
        self.bias_posterior_fn = bias_posterior_fn
        self.bias_divergence_fn = bias_divergence_fn


    def build(self):

        self.kernel_posterior = Parameter(self.kernel_posterior_fn(
            self.inputs,
            self.outputs,
            **self.factory_kwargs
        )

        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(
                self.inputs,
                self.outputs,
                **self.factory_kwargs
            )

        self.bias_prior = self.bias_prior()
        self.bias_posterior = self.bias_posterior()


    def forward(self, inputs):
        return means, stddevs, samples


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
        build()


    def build():

        self.linear_activation = LeakyReLU(negative_slope=0.2)
        self.aleatoric_activation = SoftPlus()

        self.common_layers = Linear(self.n_inputs, self.n_hidden)
        self.special_layers = ModuleDict()
        self.epistemic_layers = ModuleDict()
        self.aleatoric_layers = ModuleDict()
        for ii in range(n_outputs):
            self.special_layers.update({f'specialized{ii}': Linear(self.n_hidden, self.n_special[ii])})
            self.epistemic_layers.update({f'model{ii}': EpistemicLayer(self.n_special[ii], 1)})
            self.aleatoric_layers.update({f'noise{ii}': Linear(self.n_special[ii], 1)})


    def forward(self, inputs):

        outputs = [None] * (2 * self.n_outputs)

        commons = self.common_layers(inputs)
        commons = self.linear_activation(commons)

        for ii in range(self.n_outputs):

            specials = self.special_layers[f'specialized{ii}'](commons)
            epistemic_means, epistemic_stddevs, sample_means = self.epistemic_layers[f'model{ii}'](specials)
            aleatoric_stddevs = self.aleatoric_layers[f'noise{ii}'](specials)

            model_dist = tnd.normal.Normal(epistemic_means, epistemic_stddevs)
            noise_dist = tnd.normal.Normal(sample_means, aleatoric_stddevs)

            outputs[2*ii] = tnd.independent.Independent(model_dist, 1)
            outputs[2*ii+1] = tnd.independent.Independent(noise_dist, 1)

        return outputs
