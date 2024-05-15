import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Lambda, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl


# Helper function to generate output distributions
def create_normal_distributions(moments):
    return tfd.Normal(loc=moments[..., :1], scale=moments[..., 1:])


# Required custom class for variational layer
class DenseEpistemicLayer(tfpl.DenseReparameterization):

    def __init__(self, *args, **kwargs):
        super(DenseEpistemicLayer, self).__init__(*args, **kwargs)

    def _compute_mean_distribution_moments(self, inputs):
        kernel_mean = self.kernel_posterior.mean()
        kernel_stddev = self.kernel_posterior.stddev()
        bias_mean = self.bias_posterior.mean()
        dist_mean = tf.matmul(inputs, kernel_mean) + bias_mean
        dist_var = tf.matmul(inputs ** 2, kernel_stddev ** 2)
        dist_stddev = tf.sqrt(dist_var)
        return dist_mean, dist_stddev

    def call(self, inputs):
        samples = super(DenseEpistemicLayer, self).call(inputs)
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return means, stddevs, samples


#Model architecture - should this be a class inheriting Model instead?
def create_model(n_inputs, n_hidden, n_outputs, n_specialized=None, verbosity=0):

    leaky_relu = LeakyReLU(alpha=0.2)

    n_special = [n_hidden] * n_outputs
    if isinstance(n_specialized, (list, tuple)):
        for ii in range(n_outputs):
            n_special[ii] = n_specialized[ii] if ii < len(n_specialized) else n_specialized[-1]

    inputs = Input(shape=(n_inputs,))
    commons = Dense(n_hidden, activation=leaky_relu)(inputs)

    outputs = [None] * (2 * n_outputs)

    for ii in range(n_outputs):

        specials = Dense(n_special[ii], activation=leaky_relu, name=f'specialized{ii}')(commons)

        #variational_object = DenseEpistemicLayer(1, name=f'epistemic{ii}')
        epistemic_means, epistemic_stddevs, sample_means = DenseEpistemicLayer(1, name=f'model{ii}')(specials)
        aleatoric_stddevs = Dense(1, activation='softplus', name=f'noise{ii}')(specials)

        epistemics = Concatenate(name=f'epistemic{ii}')([epistemic_means, epistemic_stddevs])
        aleatorics = Concatenate(name=f'aleatoric{ii}')([sample_means, aleatoric_stddevs])

        model_dist = tfpl.DistributionLambda(
            make_distribution_fn=create_normal_distributions,
            name=f'model_output{ii}'
        )(epistemics)
        noise_dist = tfpl.DistributionLambda(
            make_distribution_fn=create_normal_distributions,
            name=f'noise_output{ii}'
        )(aleatorics)

        # These are distributions, not tensors
        outputs[2*ii] = model_dist
        outputs[2*ii+1] = noise_dist

    return Model(inputs, outputs)

