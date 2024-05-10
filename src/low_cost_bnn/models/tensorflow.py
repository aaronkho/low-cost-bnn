import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

# Required conversion function
def mean_dist_fn(variational_layer):

    def mean_dist(inputs):
        bias_mean = variational_layer.bias_posterior.mean()
        kernel_mean = variational_layer.kernel_posterior.mean()
        kernel_std = variational_layer.kernel_posterior.stddev()
        mu_mean = tf.matmul(inputs, kernel_mean) + bias_mean
        mu_var = tf.matmul(inputs ** 2, kernel_std ** 2)
        mu_std = tf.sqrt(mu_var)
        return tfd.Normal(mu_mean, mu_std)

    return mean_dist

#Model architecture - should this be a class inheriting Model instead?
def create_model(n_inputs, n_hidden, n_outputs, n_specialized=None):

    leaky_relu = LeakyReLU(negative_slope=0.2)

    n_special = [n_hidden] * n_outputs
    if isinstance(n_specialized, (list, tuple)):
        for ii in range(n_outputs):
            n_special[ii] = n_specialized[ii] if ii < len(n_specialized) else n_specialized[-1]

    input_layer = Input(shape=(n_inputs, ))
    hidden_layer = Dense(n_hidden, activation=leaky_relu)(input_layer)

    specialized_layers = [None] * n_outputs
    variational_objects = [None] * n_outputs
    aleatoric_layers = [None] * n_outputs
    epistemic_layers = [None] * n_outputs

    output_layer = [None] * (2 * n_outputs)

    for ii in range(n_outputs):

        variational_objects[ii] = tfpl.DenseReparameterization(1, name=f'mu{ii}')
        specialized_layers[ii] = Dense(n_special[ii], activation=leaky_relu)(hidden_layer)
        aleatoric_layers[ii] = Dense(1, activation='softplus', name=f'sigma{ii}')(specialized_layers[ii])
        epistemic_layers[ii] = variational_objects[ii](specialized_layers[ii])

        model_dist = tfpl.DistributionLambda(mean_dist_fn(variational_objects[ii]), name=f'output{ii}')(specialized_layers[ii])
        noise_dist = tfpl.DistributionLambda(lambda p: tfd.Normal(p[0],p[1]))((epistemic_layers[ii], aleatoric_layers[ii]))

        output_layer[2*ii] = model_dist[ii]
        output_layer[2*ii+1] = noise_dist[ii]

    return Model(input_layer, output_layer)

