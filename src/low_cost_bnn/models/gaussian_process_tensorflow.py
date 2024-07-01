import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import tensorflow_models as tfm
from ..utils.helpers_tensorflow import default_dtype



# ------ LAYERS ------


# Taken from tf-models-official and modified
class RandomFeatureGaussianProcess(tf.keras.layers.Layer):


    def __init__(
        self,
        units,
        num_inducing=1024,
        gp_kernel_type='gaussian',
        gp_kernel_scale=1.0,
        gp_output_bias=0.0,
        gp_kernel_scale_trainable=False,
        gp_output_bias_trainable=False,
        gp_cov_momentum=-1,
        gp_cov_ridge_penalty=1.0,
        scale_random_features=True,
        custom_random_features_initializer=None,
        custom_random_features_activation=None,
        l2_regularization=0.0,
        gp_cov_likelihood='binary_logistic',
        return_gp_cov=True,
        return_random_features=False,
        dtype=None,
        name='random_feature_gaussian_process',
        **gp_output_kwargs
    ):

        super().__init__(name=name, dtype=dtype)
        self.units = units
        self.num_inducing = num_inducing

        self.gp_input_scale = 1. / tf.sqrt(gp_kernel_scale)
        self.gp_feature_scale = tf.sqrt(2. / float(num_inducing))

        self.scale_random_features = scale_random_features
        self.return_random_features = return_random_features
        self.return_gp_cov = return_gp_cov

        self.gp_kernel_type = gp_kernel_type
        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias
        self.gp_kernel_scale_trainable = gp_kernel_scale_trainable
        self.gp_output_bias_trainable = gp_output_bias_trainable

        self.use_custom_random_features = use_custom_random_features
        self.custom_random_features_initializer = custom_random_features_initializer
        self.custom_random_features_activation = custom_random_features_activation

        self.l2_regularization = l2_regularization
        self.gp_output_kwargs = gp_output_kwargs

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_likelihood = gp_cov_likelihood

        if self.use_custom_random_features:
            # Default to Gaussian RBF kernel
            self.random_features_bias_initializer = tf.random_uniform_initializer(minval=0., maxval=2. * math.pi)
            if self.custom_random_features_initializer is None:
                self.custom_random_features_initializer = (tf.keras.initializers.RandomNormal(stddev=1.))
            if self.custom_random_features_activation is None:
                self.custom_random_features_activation = tf.math.cos


    def build(self, input_shape):

        self._random_feature = self._make_random_feature_layer(name='gp_random_feature')
        self._random_feature.build(input_shape)
        input_shape = self._random_feature.compute_output_shape(input_shape)

        if self.return_gp_cov:
            self._gp_cov_layer = LaplaceRandomFeatureCovariance(
                momentum=self.gp_cov_momentum,
                ridge_penalty=self.gp_cov_ridge_penalty,
                likelihood=self.gp_cov_likelihood,
                dtype=self.dtype,
                name='gp_covariance'
            )
            self._gp_cov_layer.build(input_shape)

        self._gp_output_layer = tf.keras.layers.Dense(
            units=self.units,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            dtype=self.dtype,
            name='gp_output_weights',
            **self.gp_output_kwargs
        )
        self._gp_output_layer.build(input_shape)

        self._gp_output_bias = tf.Variable(
            initial_value=[self.gp_output_bias] * self.units,
            dtype=self.dtype,
            trainable=self.gp_output_bias_trainable,
            name='gp_output_bias'
        )

        self.built = True


    def _make_random_feature_layer(self, name):
        # Always use RandomFourierFeatures layer from tf.keras.
        return tf.keras.layers.experimental.RandomFourierFeatures(
            output_dim=self.num_inducing,
            kernel_initializer=self.gp_kernel_type,
            scale=self.gp_kernel_scale,
            trainable=self.gp_kernel_scale_trainable,
            dtype=self.dtype,
            name=name
        )


    def reset_covariance_matrix(self):
        # Required at the beginning of every epoch!
        self._gp_cov_layer.reset_precision_matrix()


    def call(self, inputs, global_step=None, training=None):

        gp_input_scale = tf.cast(self.gp_input_scale, inputs.dtype)
        gp_inputs = inputs * gp_input_scale

        gp_feature = self._random_feature(gp_inputs)

        if self.scale_random_features:
            gp_feature_scale = tf.cast(self.gp_feature_scale, inputs.dtype)
            gp_feature = gp_feature * gp_feature_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if self.return_gp_cov:
            gp_covmat = self._gp_cov_layer(gp_feature, gp_output, training)

        # Assembles model output.
        model_output = [gp_output,]
        if self.return_gp_cov:
            model_output.append(gp_covmat)
        if self.return_random_features:
            model_output.append(gp_feature)

        return model_output



# Taken from tf-models-official and modified
class LaplaceRandomFeatureCovariance(tf.keras.layers.Layer):


    _SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')


    def __init__(
        self,
        momentum=0.999,
        ridge_penalty=1.,
        likelihood='binary_logistic',
        dtype=None,
        name='laplace_covariance'
    ):

        if likelihood not in self._SUPPORTED_LIKELIHOOD:
            raise ValueError(f'"likelihood" must be one of {self._SUPPORTED_LIKELIHOOD}, got {likelihood}.')

        super().__init__(dtype=dtype, name=name)

        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.likelihood = likelihood


    def compute_output_shape(self, input_shape):
        gp_feature_dim = input_shape[-1]
        return tf.TensorShape([gp_feature_dim, gp_feature_dim])


    def build(self, input_shape):

        gp_feature_dim = input_shape[-1]

        # Posterior precision matrix for the GP's random feature coefficients
        self.initial_precision_matrix = (self.ridge_penalty * tf.eye(gp_feature_dim, dtype=self.dtype))

        self.precision_matrix = self.add_weight(
            name='gp_precision_matrix',
            shape=(gp_feature_dim, gp_feature_dim),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Identity(self.ridge_penalty),
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

        self.built = True


    def make_precision_matrix_update_op(self, gp_feature, logits, precision_matrix):

        batch_size = tf.shape(gp_feature)[0]
        batch_size = tf.cast(batch_size, dtype=gp_feature.dtype)

        # Computes batch-specific normalized precision matrix
        if self.likelihood == 'binary_logistic':
            prob = tf.sigmoid(logits)
            prob_multiplier = prob * (1. - prob)
            if logits.shape[-1] > 1:
                prob_multiplier = tf.expand_dims(tf.math.reduce_max(prob_multiplier, axis=-1), axis=-1)
        elif self.likelihood == 'poisson':
            prob_multiplier = tf.exp(logits)
            if logits.shape[-1] > 1:
                prob_multiplier = tf.expand_dims(tf.math.reduce_max(prob_multiplier, axis=-1), axis=-1)
        else:
            prob_multiplier = tf.constant(1.0, shape=tf.shape(gp_feature))

        gp_feature_adjusted = tf.sqrt(prob_multiplier) * gp_feature
        precision_matrix_minibatch = tf.matmul(gp_feature_adjusted, gp_feature_adjusted, transpose_a=True)

        # Updates the population-wise precision matrix
        if self.momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision matrices
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.momentum * precision_matrix + (1. - self.momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum
            # Only pass data through once if using this option
            precision_matrix_new = precision_matrix + precision_matrix_minibatch

        return precision_matrix.assign(precision_matrix_new)


    def reset_precision_matrix(self):
        precision_matrix_reset_op = self.precision_matrix.assign(self.initial_precision_matrix)
        self.add_update(precision_matrix_reset_op)


    def compute_predictive_covariance(self, gp_feature):

        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = tf.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = tf.matmul(feature_cov_matrix, gp_feature, transpose_b=True) * self.ridge_penalty
        gp_cov_matrix = tf.matmul(gp_feature, cov_feature_product)

        return gp_cov_matrix


    def _get_training_value(self, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        return training


    def call(self, inputs, logits=None, training=None):

        batch_size = tf.shape(inputs)[0]
        training = self._get_training_value(training)

        if training:
            precision_matrix_update_op = self.make_precision_matrix_update_op(
                gp_feature=inputs,
                logits=logits,
                precision_matrix=self.precision_matrix
            )
            # Return null during training
            self.add_update(precision_matrix_update_op)
            return tf.eye(batch_size, dtype=self.dtype)
        else:
            # Return covariance estimate during prediction
            return self.compute_predictive_covariance(gp_feature=inputs)



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

        super().__init__(**kwargs)

        self.units = units
        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units

        # Internal RandomFourierFeatures returns cos(W * h + B), scale_random_features multiplies by sqrt(2 / D)
        # Internal Dense layer acts as the trainable beta vector
        self._gaussian_layer = tfm.nlp.layers.RandomFeatureGaussianProcess(
            self.units,
            num_inducing=1024,
            gp_cov_momentum=-1,
            gp_cov_ridge_penalty=1.0,
            scale_random_features=True,
            l2_regularization=0.0,
            gp_cov_likelihood='binary_logistic',
            return_gp_cov=True,
            return_random_feature=False,
            dtype=self.dtype,
            name='rfgp'
        )
        #self._gaussian_layer = RandomFeatureGaussianProcess(...)


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
        probs = tf.nn.softmax(adjusted_logits, axis=-1) if self.units > 1 else tf.math.sigmoid(adjusted_logits)
        prediction = tf.math.argmax(probs, axis=-1)
        indices = tf.concat([tf.reshape(tf.range(outputs.shape[0]), shape=prediction.shape), prediction], axis=-1)
        uncertainty = tf.reshape(1.0 - 2.0 * tf.abs(tf.gather_nd(probs, indices=indices) - 0.5), shape=prediction.shape)
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
        base_config = super().get_config()
        config = {
            'units': self.units,
        }
        return {**base_config, **config}



# ------ LOSSES ------


CrossEntropyLoss = tf.keras.losses.SparseCategoricalCrossentropy


