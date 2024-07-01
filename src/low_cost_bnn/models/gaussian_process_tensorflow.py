import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import tensorflow_models as tfm
from ..utils.helpers_tensorflow import default_dtype



# ------ LAYERS ------


class DenseReparameterizationGaussianProcess(tf.keras.layers.Layer):


    _map = {
        'logits': 0,
        'variance': 1
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma': 1
    }
    _n_recast_params = len(_recast_map)


    def __init__(self, units, **kwargs):

        super(DenseReparameterizationGaussianProcess, self).__init__(**kwargs)

        self.units = units
        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units

        self._gaussian_layer = tfm.nlp.layers.RandomFeatureGaussianProcess(
            self.units,
            num_inducing=1024,
            normalize_input=False,
            scale_random_features=True,
            gp_cov_momentum=-1,
            custom_random_features_initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
            custom_random_features_activation=tf.math.cos,
            dtype=self.dtype,
            name='rfgp'
        )


    # Output: Shape(batch_size, n_outputs)
    @tf.function
    def call(self, inputs):
        logits, covmat = self._gaussian_layer(inputs)
        input_variance = [tf.linalg.diag_part(covmat)]
        while len(input_variance) < logits.shape[-1]:
            input_variance.append(input_variance[0])
        variance = tf.stack(input_variance, axis=-1)
        return tf.concat([logits, variance], axis=-1)


    # Output: Shape(batch_size, batch_size)
    @tf.function
    def get_covariance(self, inputs):
        logits, covmat = self._gaussian_layer(inputs)
        return covmat


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def recast_to_prediction_epistemic(self, outputs):
        logits = tf.gather(outputs, indices=[0], axis=-1)
        variance = tf.gather(outputs, indices=[1], axis=-1)
        adjusted_logits = logits / tf.sqrt(1.0 + (tf.math.acos(1.0) / 8.0) * variance)
        prediction = tf.gather(tf.nn.softmax(adjusted_logits, axis=-1), indices=[0], axis=-1)
        uncertainty = prediction * (1.0 - prediction)
        return tf.concat([prediction, uncertainty], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic(outputs)


    def reset_covariance_matrix(self):
        self._gaussian_layer.reset_covariance_matrix()


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], self._n_outputs])


    def get_config(self):
        base_config = super(DenseReparameterizationEpistemic, self).get_config()
        config = {
            'units': self.units,
        }
        return {**base_config, **config}



# ------ LOSSES ------


CrossEntropyLoss = tf.keras.losses.SparseCategoricalCrossentropy


