import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow_probability import distributions as tfd



# ------ LAYERS ------


class DenseReparameterizationNormalInverseGamma(tf.keras.layers.Layer):


    _map = {
        'gamma': 0,
        'nu': 1,
        'alpha': 2,
        'beta': 3
    }
    _n_params = len(_map)
    _n_recast_params = 3


    def __init__(self, units, **kwargs):

        super(DenseReparameterizationNormalInverseGamma, self).__init__(**kwargs)

        self.units = units
        self.dense = Dense(self._n_params * self.units, activation=None)

        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units


    # Output: Shape(batch_size, n_outputs)
    @tf.function
    def call(self, inputs):
        outputs = self.dense(inputs)
        gamma, lognu, logalpha, logbeta = tf.split(outputs, len(self._map), axis=-1)
        nu = tf.nn.softplus(lognu)
        alpha = tf.nn.softplus(logalpha) + 1
        beta = tf.nn.softplus(logbeta)
        return tf.concat([gamma, nu, alpha, beta], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def recast_to_prediction_epistemic_aleatoric(self, outputs):
        gamma_indices = [ii for ii in range(self._map['gamma'] * self.units, self._map['gamma'] * self.units + self.units + 1)]
        nu_indices = [ii for ii in range(self._map['nu'] * self.units, self._map['nu'] * self.units + self.units + 1)]
        alpha_indices = [ii for ii in range(self._map['alpha'] * self.units, self._map['alpha'] * self.units + self.units + 1)]
        beta_indices = [ii for ii in range(self._map['beta'] * self.units, self._map['beta'] * self.units + self.units + 1)]
        prediction = tf.gather(output, indices=gamma_indices, axis=-1)
        ones = tf.ones(tf.shape(prediction), dtype=output.dtype)
        aleatoric = tf.math.divide(tf.gather(output, indices=beta_indices, axis=-1), tf.math.subtract(tf.gather(output, indices=alpha_indices, axis=-1), ones))
        epistemic = tf.math.divide(aleatoric, tf.gather(output, indices=nu_indices, axis=-1))
        return tf.concat([prediction, epistemic, aleatoric], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic_aleatoric(outputs)


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], self._n_outputs])


    def get_config(self):
        base_config = super(DenseReparameterizationNormalInverseGamma, self).get_config()
        config = {
            'units': self.units
        }
        return {**base_config, **config}



# ------ LOSSES ------


class NLLNormalInverseGammaLoss(tf.keras.losses.Loss):


    def __init__(self, weight=1.0, **kwargs):

        super(NLLNormalInverseGammaLoss, self).__init__(**kwargs)

        self.weight = weight if isinstance(weight, float) else 1.0


    @tf.function
    def call(self, target_values, distribution_moments):
        weight = tf.constant(self.weight, dtype=self.dtype)
        gammas, nus, alphas, betas = tf.unstack(distribution_moments, axis=-1)
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



class EvidenceRegularizationLoss(tf.keras.losses.Loss):


    def __init__(self, weight=1.0, **kwargs):

        super(EvidenceRegularizationLoss, self).__init__(**kwargs)

        self.weight = weight if isinstance(weight, float) else 1.0


    @tf.function
    def call(self, target_values, distribution_moments):
        weight = tf.constant(self.weight, dtype=self.dtype)
        gammas, nus, alphas, betas = tf.unstack(distribution_moments, axis=-1)
        loss = tf.math.abs(target_values - gammas) * (2.0 * nus + alphas)
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss, axis=0)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss, axis=0)
        return loss


    def get_config(self):
        base_config = super(EvidenceRegularizationLoss, self).get_config()
        config = {
            'weight': self.weight
        }
        return {**base_config, **config}



class EvidentialLoss(tf.keras.losses.Loss):


    def __init__(self, **kwargs):

        super(EvidentialLoss, self).__init__(**kwargs)



class MultiOutputEvidentialLoss(tf.keras.losses.Loss):


    def __init__(self, **kwargs):

        super(EvidentialLoss, self).__init__(**kwargs)


