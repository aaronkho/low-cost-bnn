import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import tensorflow_models as tfm
from ..utils.helpers_tensorflow import default_dtype



# ------ LAYERS ------


class DenseReparameterizationGaussianProcess(tf.keras.layers.Layer):


    _map = {
        'logits': 0
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
            dtype=self.dtype,
            name='rfgp'
        )


    # Output: Shape(batch_size, n_outputs), Shape(batch_size, batch_size)
    @tf.function
    def call(self, inputs, return_covmat=False):
        logits, covmat = self._gaussian_layer(inputs)
        return logits, covmat if return_covmat else logits


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def recast_to_prediction_epistemic(self, logits, covmat):
        adjusted_logits = tfm.nlp.layers.gaussian_process.mean_field_logits(logits, covmat, mean_field_factor=(np.pi / 8.0))
        prediction = tf.nn.softmax(adjusted_logits, axis=-1)[:, 0]
        uncertainty = prediction * (1.0 - prediction)
        return tf.concat([prediction, uncertainty], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic(outputs)


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


