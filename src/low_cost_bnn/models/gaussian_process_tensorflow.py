import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import tensorflow_models as tfm
from ..utils.helpers_tensorflow import default_dtype, default_device



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
        use_custom_random_features=True,
        custom_random_features_initializer=None,
        custom_random_features_activation=None,
        l2_regularization=0.0,
        gp_cov_likelihood='binary_logistic',
        return_gp_cov=True,
        return_random_features=False,
        name='random_feature_gaussian_process',
        dtype=None,
        **gp_output_kwargs
    ):

        super().__init__(name=name, dtype=dtype)
        self.units = units
        self.num_inducing = num_inducing

        self.gp_input_scale = 1.0 / tf.sqrt(gp_kernel_scale)
        self.gp_feature_scale = tf.sqrt(2.0 / float(num_inducing))

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
        self.random_features_bias_initializer = None

        self.l2_regularization = l2_regularization
        self.gp_output_kwargs = gp_output_kwargs

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_likelihood = gp_cov_likelihood

        if self.use_custom_random_features:
            # Default to Gaussian RBF kernel
            self.random_features_bias_initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=2. * math.pi)
            if self.custom_random_features_initializer is None:
                self.custom_random_features_initializer = tf.keras.initializers.RandomNormal(stddev=1.)
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
                name='gp_covariance',
                dtype=self.dtype
            )
            self._gp_cov_layer.build(input_shape)

        self._gp_output_layer = tf.keras.layers.Dense(
            units=self.units,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name='gp_output_weights',
            dtype=self.dtype,
            **self.gp_output_kwargs
        )
        self._gp_output_layer.build(input_shape)

        self._gp_output_bias = tf.Variable(
            initial_value=[self.gp_output_bias] * self.units,
            trainable=self.gp_output_bias_trainable,
            name='gp_output_bias',
            dtype=self.dtype,
        )

        self.built = True


    def _make_random_feature_layer(self, name):
        if self.use_custom_random_features:
            return Dense(
                units=self.num_inducing,
                use_bias=True,
                activation=self.custom_random_features_activation,
                kernel_initializer=self.custom_random_features_initializer,
                bias_initializer=self.random_features_bias_initializer,
                trainable=True,
                name=name
            )
        else:
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

        gp_input_scale = tf.cast(self.gp_input_scale, self.dtype)
        gp_inputs = inputs * gp_input_scale

        gp_feature = self._random_feature(gp_inputs)

        if self.scale_random_features:
            gp_feature_scale = tf.cast(self.gp_feature_scale, self.dtype)
            gp_feature = gp_feature * gp_feature_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if self.return_gp_cov:
            gp_covmat = self._gp_cov_layer(gp_feature, gp_output, training)

        # Assembles model output.
        model_output = [gp_output]
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
        name='laplace_covariance',
        **kwargs
    ):

        if likelihood not in self._SUPPORTED_LIKELIHOOD:
            raise ValueError(f'"likelihood" must be one of {self._SUPPORTED_LIKELIHOOD}, got {likelihood}.')

        super().__init__(name=name, **kwargs)

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
        batch_size = tf.cast(batch_size, dtype=self.dtype)

        # Computes batch-specific normalized precision matrix
        if self.likelihood == 'binary_logistic':
            prob = tf.sigmoid(logits)
            prob_multiplier = prob * (1.0 - prob)
            if logits.shape[-1] > 1:
                prob_multiplier = tf.expand_dims(tf.math.reduce_max(prob_multiplier, axis=-1), axis=-1)
        elif self.likelihood == 'poisson':
            prob_multiplier = tf.exp(logits)
            if logits.shape[-1] > 1:
                prob_multiplier = tf.expand_dims(tf.math.reduce_max(prob_multiplier, axis=-1), axis=-1)
        else:
            prob_multiplier = tf.constant(1.0, shape=tf.shape(gp_feature), dtype=self.dtype)

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


    def __init__(self, units, threshold=0.5, **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units
        self._threshold = float(threshold) if isinstance(threshold, float) else 0.5

        # Internal RandomFourierFeatures returns cos(W * h + B), scale_random_features multiplies by sqrt(2 / D)
        # Internal Dense layer acts as the trainable beta vector
        #self._gaussian_layer = tfm.nlp.layers.RandomFeatureGaussianProcess(
        self._gaussian_layer = RandomFeatureGaussianProcess(
            self.units,
            num_inducing=1024,
            gp_cov_momentum=-1,
            gp_cov_ridge_penalty=1.0,
            scale_random_features=True,
            l2_regularization=0.0,
            gp_cov_likelihood='binary_logistic',
            return_gp_cov=True,
            return_random_features=False,
            name='rfgp',
            dtype=self.dtype
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
        logit_indices = [ii for ii in range(self._map['logits'] * self.units, self._map['logits'] * self.units + self.units)]
        variance_indices = [ii for ii in range(self._map['variance'] * self.units, self._map['variance'] * self.units + self.units)]
        logits = tf.gather(outputs, indices=logit_indices, axis=-1)
        variance = tf.gather(outputs, indices=variance_indices, axis=-1)
        mean_field_logits = tf.math.divide(logits, tf.sqrt(1.0 + tf.math.multiply(tf.math.acos(tf.constant([1.0], dtype=self.dtype)) / 8.0, variance)))
        if self.units > 1:
            full_probabilities = tf.nn.softmax(mean_field_logits, axis=-1)
            maximum_index = tf.math.argmax(full_probabilities, axis=-1)
            predicted_class_mask = tf.one_hot(maximum_index, depth=self.units, axis=-1)
            probabilities = tf.reduce_sum(tf.math.multiply(full_probabilities, predicted_class_mask), axis=-1)
            prediction = tf.cast(maximum_index, dtype=self.dtype)
        else:
            probabilities = tf.squeeze(tf.math.sigmoid(mean_field_logits), axis=-1)
            threshold_shift = 0.5 * tf.ones(shape=tf.shape(probabilities), dtype=self.dtype) - self._threshold
            prediction = tf.math.round(probabilities + threshold_shift)
        ones = tf.ones(tf.shape(probabilities), dtype=self.dtype)
        uncertainty = tf.subtract(ones, tf.abs(tf.math.subtract(tf.math.add(probabilities, probabilities), ones)))
        return tf.stack([prediction, uncertainty], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic(outputs)


    def reset_covariance_matrix(self):
        self._gaussian_layer.reset_covariance_matrix()


    @property
    def threshold(self):
        return self._threshold


    @threshold.setter
    def threshold(self, val):
        if isinstance(val, (float, int)):
            self._threshold = float(val)


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], self._n_outputs])


    def get_config(self):
        base_config = super().get_config()
        config = {
            'units': self.units,
            'threshold': self.threshold,
        }
        return {**base_config, **config}



# ------ LOSSES ------


class CrossEntropyLoss(tf.keras.losses.Loss):


    def __init__(
        self,
        entropy_weight=1.0,
        name='crossentropy',
        reduction='sum',
        dtype=default_dtype,
        **kwargs
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype

        self._entropy_weight = entropy_weight
        self._entropy_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, name=self.name+'_binary', reduction=self.reduction)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_entropy_loss(self, targets, predictions):
        weight = tf.constant(self._entropy_weight, dtype=self.dtype)
        base = tf.cast(self._entropy_loss_fn(targets, predictions), dtype=self.dtype)
        loss = weight * base
        return loss


    @tf.function
    def call(self, targets, predictions):
        entropy_loss = self._calculate_entropy_loss(targets, predictions)
        total_loss = entropy_loss
        return total_loss


    def get_config(self):
        base_config = super().get_config()
        config = {
            'entropy_weight': self._entropy_weight
        }
        return {**base_config, **config}



class MultiClassCrossEntropyLoss(tf.keras.losses.Loss):


    def __init__(
        self,
        entropy_weight=1.0,
        name='crossentropy',
        reduction='sum',
        dtype=default_dtype,
        **kwargs
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype

        self._entropy_weight = entropy_weight
        self._entropy_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name=self.name+'_categorical', reduction=self.reduction)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_entropy_loss(self, targets, predictions):
        weight = tf.constant(self._entropy_weight, dtype=self.dtype)
        base = tf.cast(self._entropy_loss_fn(targets, predictions), dtype=self.dtype)
        loss = weight * base
        return loss


    @tf.function
    def call(self, targets, predictions):
        entropy_loss = self._calculate_entropy_loss(targets, predictions)
        total_loss = entropy_loss
        return total_loss


    def get_config(self):
        base_config = super().get_config()
        config = {
            'entropy_weight': self._entropy_weight
        }
        return {**base_config, **config}



class MultiOutputCrossEntropyLoss(tf.keras.losses.Loss):


    def __init__(
        self,
        n_outputs,
        entropy_weights,
        name='multi_crossentropy',
        reduction='sum',
        dtype=default_dtype,
        **kwargs
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._entropy_weights = [0.0] * self.n_outputs
        for ii in range(self.n_outputs):
            ent_w = 1.0
            if isinstance(entropy_weights, (list, tuple)):
                ent_w = entropy_weights[ii] if ii < len(entropy_weights) else entropy_weights[-1]
            self._loss_fns[ii] = CrossEntropyLoss(
                ent_w,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                dtype=self.dtype
            )
            self._entropy_weights[ii] = ent_w


    # Input
    @tf.function
    def _calculate_entropy_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_entropy_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    def call(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii](target_stack[ii], prediction_stack[ii]))
        total_loss = tf.stack(losses, axis=-1)
        if self.reduction == 'mean':
            total_loss = tf.reduce_mean(total_loss)
        elif self.reduction == 'sum':
            total_loss = tf.reduce_sum(total_loss)
        return total_loss


    def get_config(self):
        base_config = super().get_config()
        config = {
        }
        return {**base_config, **config}



class MultiOutputMultiClassCrossEntropyLoss(tf.keras.losses.Loss):


    def __init__(
        self,
        n_outputs,
        entropy_weights,
        name='multi_crossentropy',
        reduction='sum',
        dtype=default_dtype,
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._entropy_weights = [0.0] * self.n_outputs
        for ii in range(self.n_outputs):
            ent_w = 1.0
            if isinstance(entropy_weights, (list, tuple)):
                ent_w = entropy_weights[ii] if ii < len(entropy_weights) else entropy_weights[-1]
            self._loss_fns[ii] = MultiClassCrossEntropyLoss(
                ent_w,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                dtype=self.dtype
            )
            self._entropy_weights[ii] = ent_w


    # Input
    @tf.function
    def _calculate_entropy_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_entropy_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    def call(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii](target_stack[ii], prediction_stack[ii]))
        total_loss = tf.stack(losses, axis=-1)
        if self.reduction == 'mean':
            total_loss = tf.reduce_mean(total_loss)
        elif self.reduction == 'sum':
            total_loss = tf.reduce_sum(total_loss)
        return total_loss


    def get_config(self):
        base_config = super().get_config()
        config = {
        }
        return {**base_config, **config}

