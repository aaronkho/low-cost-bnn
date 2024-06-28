import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow_models.nlp.layers import RandomFeatureGaussianProcess
from ..utils.helpers_tensorflow import default_dtype



# ------ LAYERS ------


class SpectralNormalizedGaussianModel(tf.keras.models.Model):


    def __init__(self, units, **kwargs):

        super(SpectralNormalizedGaussianModel, self).__init__(**kwargs)

        self.units = units

        self._gaussian_layer = RandomProcessGaussianProcess(self.units)


    # Output: Shape(batch_size, n_outputs)
    @tf.function
    def call(self, inputs):
        outputs = self._dense(inputs)
        gamma, lognu, logalpha, logbeta = tf.split(outputs, len(self._map), axis=-1)
        nu = tf.nn.softplus(lognu)
        alpha = tf.nn.softplus(logalpha) + 1
        beta = tf.nn.softplus(logbeta)
        return tf.concat([gamma, nu, alpha, beta], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def recast_to_prediction_epistemic_aleatoric(self, outputs):
        gamma_indices = [ii for ii in range(self._map['gamma'] * self.units, self._map['gamma'] * self.units + self.units)]
        nu_indices = [ii for ii in range(self._map['nu'] * self.units, self._map['nu'] * self.units + self.units)]
        alpha_indices = [ii for ii in range(self._map['alpha'] * self.units, self._map['alpha'] * self.units + self.units)]
        beta_indices = [ii for ii in range(self._map['beta'] * self.units, self._map['beta'] * self.units + self.units)]
