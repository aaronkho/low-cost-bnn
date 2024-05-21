import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, LeakyReLU
from tensorflow_probability import distributions as tfd


def create_student_t_posterior(gamma, nu, alpha, beta, verbosity=0):
    mean = gamma
    sigma = tf.sqrt(beta * (1.0 + nu) / (nu * alpha))
    dof = 2.0 * alpha
    return tfd.StudentT(df=dof, loc=mean, scale=sigma)


class DenseReparameterizationNormalInverseGamma(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(DenseReparameterizationNormalInverseGamma, self).__init__(**kwargs)
        self.units = units
        self.dense = Dense(4 * self.units, activation=None)


    def call(self, inputs):
        outputs = self.dense(inputs)
        gamma, lognu, logalpha, logbeta = tf.split(outputs, 4, axis=-1)
        nu = tf.nn.softplus(lognu)
        alpha = tf.nn.softplus(logalpha) + 1
        beta = tf.nn.softplus(logbeta)
        return tf.concat([gamma, nu, alpha, beta], axis=-1)


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], 4 * self.units])


    def get_config(self):
        base_config = super(DenseReparameterizationNormalInverseGamma, self).get_config()
        config = {
            'units': self.units
        }
        return {**base_config, **config}


class EvidentialBNN(tf.keras.models.Model):

    def __init__(self, n_output, n_hidden, n_special):
        super(EvidentialBNN, self).__init__()
        self.n_outputs = n_output
        self.n_hiddens = list(n_hidden) if isinstance(n_hidden, (list, tuple)) else [20]
        self.n_specials = list(n_special) if isinstance(n_special, (list, tuple)) else [5]
        while len(self.n_specials) < self.n_outputs:
            self.n_specials.append(self.n_specials[-1])
        if len(self.n_specials) > self.n_outputs:
            self.n_specials = self.n_specials[:self.n_outputs]

        self.base_activation = LeakyReLU(alpha=0.2)

        self.common_layers = tf.keras.Sequential()
        for ii in range(len(self.n_hidden)):
            self.common_layers.add(Dense(self.n_hidden[ii], activation=self.base_activation, name=f'common{ii}')

        self.output_channels = [None] * self.n_outputs
        for jj in range(self.n_outputs):
            channel = tf.keras.Sequential()
            channel.add(Dense(self.n_specials[jj], activation=self.base_activation, name=f'specialized{jj}')
            channel.add(DenseReparameterizationNormalInverseGamma(1, name=f'output{jj}')
            self.output_channels[jj] = channel


    def call(self, inputs):
        commons = self.common_layers(inputs)
        specials = []
        for jj in range(len(self.output_channels)):
            specials.append(self.output_channels[jj](commons))
        outputs = Concatenate(specials)
        return outputs


    def get_config(self):
        base_config = super(EvidentialBNN, self).get_config()
        config = {
            'n_output': self.n_outputs,
            'n_hidden': self.n_hiddens,
            'n_special': self.n_specials
        }
        return {**base_config, **config}


class NLLNormalInverseGammaLoss(tf.keras.losses.Loss):

    def __init__(self, weight=1.0, **kwargs):
        super(NLLNormalInverseGammaLoss, self).__init__(**kwargs)
        self.weight = weight if isinstance(weight, float) else 1.0


    def call(self, targets, gammas, nus, alphas, betas):
        weight = tf.constant(self.weight, dtype=self.dtype)
        omegas = 2.0 * betas * (1.0 + nus)
        loss = (
            0.5 * tf.math.log(np.pi / nus) -
            alphas * tf.math.log(omegas) +
            (alphas + 0.5) * tf.math.log(nus * (targets - gammas) ** 2 + omegas) +
            tf.math.lgamma(alphas) - tf.math.lgamma(alphas + 0.5)
        )
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss, axis=0)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss, axis=0)
        return loss


    def get_config(self):
        base_config = super(NLLNormalInverseGammaLoss, self).get_config()
        config = {
            'weight': self.weight
        }
        return {**base_config, **config}



def create_model(n_input, n_output, n_hidden, n_special):
    model = EvidentialBNN(n_output, n_hidden, n_special)
    model.build((None, n_input))
    return model
