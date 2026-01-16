import os
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Dense
from ..utils.helpers_tensorflow import default_dtype, default_device, get_fuzz_factor



class DenseReparameterizationZeroUncertainty(tf.keras.layers.Layer):


    _map = {
        'mu': 0,
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma': 1,
    }
    _n_recast_params = len(_recast_map)


    def __init__(self, units, **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units

        self._dense = Dense(self._n_outputs, activation=None)


    # Output: Shape(batch_size, n_outputs)
    @tf.function
    def call(self, inputs):
        return self._dense(inputs)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_zero(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.units, self._map['mu'] * self.units + self.units)])
        mean = tf.gather(outputs, indices=indices, axis=-1)
        stdev = tf.zeros_like(mean, dtype=self.dtype)
        return tf.concat([mean, stdev], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    def _recast(self, outputs):
        return self.recast_to_prediction_zero(outputs)


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], self._n_outputs])


    def get_config(self):
        base_config = super().get_config()
        config = {
            'units': self.units
        }
        return {**base_config, **config}



# ------ LOSSES ------


class SquareErrorLoss(tf.keras.losses.Loss):


    def __init__(self, name='se', dtype=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype


    # Input: Shape(batch_size, dist_moments) -> Output: Shape(batch_size)
    @tf.function
    def call(self, targets, predictions):
        loss = tf.math.pow(predictions - targets, 2)
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss)
        return loss


    def get_config(self):
        base_config = super().get_config()
        config = {
        }
        return {**base_config, **config}



class RelativeSquareErrorLoss(tf.keras.losses.Loss):


    def __init__(self, name='rse', dtype=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype

        self._fuzz = tf.constant([get_fuzz_factor(self.dtype)], dtype=self.dtype)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape(batch_size)
    @tf.function
    def call(self, targets, predictions):
        loss = tf.math.divide_no_nan(tf.math.pow(predictions - targets, 2), tf.math.pow(targets, 2) + self._fuzz)
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss)
        return loss



class MixedSquareErrorLoss(tf.keras.losses.Loss):


    def __init__(
        self,
        se_weight=1.0,
        rse_weight=1.0,
        name='mix',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs,
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype

        self._square_error_weight = se_weight
        self._relative_square_error_weight = rse_weight
        if isinstance(self._square_error_weight, (list, tuple, np.ndarray)):
            self._square_error_weight = self._square_error_weight[0]
        if isinstance(self._relative_square_error_weight, (list, tuple, np.ndarray)):
            self._relative_square_error_weight = self._relative_square_error_weight[0]
        self._square_error_loss_fn = SquareErrorLoss(name=self.name+'_se', reduction=reduction, dtype=self.dtype)
        self._relative_square_error_loss_fn = RelativeSquareErrorLoss(name=self.name+'_rse', reduction=reduction, dtype=self.dtype)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape(batch_size)
    @tf.function
    def _calculate_square_error_loss(self, targets, predictions):
        weight = tf.constant(self._square_error_weight, dtype=self.dtype)
        base = self._square_error_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape(batch_size)
    @tf.function
    def _calculate_relative_square_error_loss(self, targets, predictions):
        weight = tf.constant(self._relative_square_error_weight, dtype=self.dtype)
        base = self._relative_square_error_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments, loss_terms) -> Output: Shape(batch_size)
    @tf.function
    def call(self, targets, predictions):
        target_se_values, target_rse_values = tf.unstack(targets, axis=-1)
        prediction_se_values, prediction_rse_values = tf.unstack(predictions, axis=-1)
        rse_loss = self._calculate_square_error_loss(target_se_values, prediction_se_values)
        rrse_loss = self._calculate_relative_square_error_loss(target_rse_values, prediction_rse_values)
        total_loss = rse_loss + rrse_loss
        return total_loss


    def get_config(self):
        base_config = super().get_config()
        config = {
            'se_weight': self._square_error_weight,
            'rse_weight': self._relative_square_error_weight,
        }
        return {**base_config, **config}



class MultiOutputMixedSquareErrorLoss(tf.keras.losses.Loss):


    def __init__(
        self,
        n_outputs,
        se_weights,
        rse_weights,
        name='multi_mix',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs,
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._square_error_weights = []
        self._relative_square_error_weights = []
        for ii in range(self.n_outputs):
            se_w = 1.0
            rse_w = 1.0
            if isinstance(se_weights, (list, tuple, np.ndarray)):
                se_w = se_weights[ii] if ii < len(se_weights) else se_weights[-1]
            if isinstance(rse_weights, (list, tuple, np.ndarray)):
                rse_w = rse_weights[ii] if ii < len(rse_weights) else rse_weights[-1]
            self._loss_fns[ii] = MixedSquareErrorLoss(
                se_w,
                rse_w,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                dtype=self.dtype
            )
            self._square_error_weights.append(se_w)
            self._relative_square_error_weights.append(rse_w)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape(batch_size, n_outputs)
    @tf.function
    def _calculate_square_error_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_square_error_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape(batch_size, n_outputs)
    @tf.function
    def _calculate_relative_square_error_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_relative_square_error_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    # Input: Shape(batch_size, dist_moments, loss_terms, n_outputs) -> Output: Shape(batch_size, n_outputs)
    @tf.function
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
            'se_weights': self._square_error_weights,
            'rse_weights': self._relative_square_error_weights,
        }
        return {**base_config, **config}
