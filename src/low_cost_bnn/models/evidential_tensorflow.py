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

        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units
        self._dense = Dense(self._n_outputs, activation=None)


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
        gamma_indices = [ii for ii in range(self._map['gamma'] * self.units, self._map['gamma'] * self.units + self.units)]
        nu_indices = [ii for ii in range(self._map['nu'] * self.units, self._map['nu'] * self.units + self.units)]
        alpha_indices = [ii for ii in range(self._map['alpha'] * self.units, self._map['alpha'] * self.units + self.units)]
        beta_indices = [ii for ii in range(self._map['beta'] * self.units, self._map['beta'] * self.units + self.units)]
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


class NormalInverseGammeNLLLoss(tf.keras.losses.Loss):


    def __init__(self, **kwargs):

        super(NormalInverseGammaNLLLoss, self).__init__(**kwargs)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def call(self, target_values, distribution_moments):
        targets, _, _, _ = tf.unstack(target_values, axis=-1)
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
        base_config = super(NormalInverseGammaNLLLoss, self).get_config()
        config = {
        }
        return {**base_config, **config}



class EvidenceRegularizationLoss(tf.keras.losses.Loss):


    def __init__(self, **kwargs):

        super(EvidenceRegularizationLoss, self).__init__(**kwargs)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def call(self, target_values, distribution_moments):
        targets, _, _, _ = tf.unstack(target_values, axis=-1)
        gammas, nus, alphas, betas = tf.unstack(distribution_moments, axis=-1)
        loss = tf.math.abs(targets - gammas) * (2.0 * nus + alphas)
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss, axis=0)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss, axis=0)
        return loss


    def get_config(self):
        base_config = super(EvidenceRegularizationLoss, self).get_config()
        config = {
        }
        return {**base_config, **config}



class EvidentialLoss(tf.keras.losses.Loss):


    def __init__(self, likelihood_weight=1.0, regularization_weight=1.0, name='evidential', reduction='sum', **kwargs):

        super(EvidentialLoss, self).__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = tf.keras.backend.floatx()
        self._likelihood_weight = likelihood_weight
        self._regularization_weight = regularization_weight
        self._likelihood_loss_fn = NormalInverseGammaNLLLoss(name=self.name+'_nll', reduction=self.reduction)
        self._regularization_loss_fn = EvidenceRegularizationLoss(name=self.name+'_reg', reduction=self.reduction)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = tf.constant(self._likelihood_weight, dtype=self.dtype)
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_regularization_loss(self, targets, predictions):
        weights = tf.constant(self._regularization_weight, dtype=self.dtype)
        base = self._regularization_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    @tf.function
    def call(self, targets, predictions):
        likelihood_loss = self._calculate_likelihood_loss(targets, predictions)
        regularization_loss = self._calculate_regularization_loss(targets, predictions)
        total_loss = likelihood_loss + regularization_loss
        return total_loss


    def get_config(self):
        base_config = super(EvidentialLoss, self).get_config()
        config = {
            'likelihood_weight': self._likelihood_weight,
            'regularization_weight': self._regularization_weight
        }
        return {**base_config, **config}



class MultiOutputEvidentialLoss(tf.keras.losses.Loss):


    def __init__(self, n_outputs, likelihood_weights, regularization_weights, name='multi_evidential', reduction='sum', **kwargs):

        super(MultiOutputEvidentialLoss, self).__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = tf.keras.backend.floatx()
        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._likelihood_weights = []
        self._regularization_weights = []
        for ii in range(self.n_outputs):
            nll_w = 1.0
            reg_w = 1.0
            if isinstance(likelihood_weights, (list, tuple)):
                nll_w = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
            if isinstance(regularization_weights, (list, tuple)):
                reg_w = regularization_weights[ii] if ii < len(regularization_weights) else regularization_weights[-1]
            self._loss_fns[ii] = EvidentialLoss(nll_w, reg_w, name=f'{self.name}_out{ii}', reduction=self.reduction)
            self._likelihood_weights.append(nll_w)
            self._regularization_weights.append(reg_w)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    @tf.function
    def _calculate_likelihood_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_likelihood_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    @tf.function
    def _calculate_regularization_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_regularization_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    # Input: Shape(batch_size, dist_moments, loss_terms, n_outputs) -> Output: Shape([batch_size])
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
        base_config = super(MultiOutputEvidentialLoss, self).get_config()
        config = {
            'likelihood_weights': self._likelihood_weights,
            'regularization_weights': self._regularization_weights,
        }
        return {**base_config, **config}


