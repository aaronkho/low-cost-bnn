import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from ..utils.helpers_tensorflow import default_dtype, get_fuzz_factor



# ------ LAYERS ------


# Required custom class for variational layer
class DenseReparameterizationEpistemic(tfpl.DenseReparameterization):


    _map = {
        'mu': 0,
        'sigma': 1,
        'sample': 2
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma': 1
    }
    _n_recast_params = len(_recast_map)


    def __init__(self, units, **kwargs):

        super().__init__(units, **kwargs)

        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units


    def _compute_mean_distribution_moments(self, inputs):
        kernel_mean = self.kernel_posterior.mean()
        kernel_stddev = self.kernel_posterior.stddev()
        bias_mean = self.bias_posterior.mean()
        dist_mean = tf.matmul(inputs, kernel_mean) + bias_mean
        dist_var = tf.matmul(inputs ** 2, kernel_stddev ** 2)
        dist_stddev = tf.sqrt(dist_var)
        return dist_mean, dist_stddev


    # Output: Shape(batch_size, n_outputs)
    @tf.function
    def call(self, inputs):
        samples = super().call(inputs)
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return tf.concat([means, stddevs, samples], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def recast_to_prediction_epistemic(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.units, self._map['mu'] * self.units + self.units)])
        indices.extend([ii for ii in range(self._map['sigma'] * self.units, self._map['sigma'] * self.units + self.units)])
        return tf.gather(outputs, indices=indices, axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic(outputs)


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], self._n_outputs])


    def get_config(self):
        base_config = super().get_config()
        config = {
        }
        return {**base_config, **config}



class DenseReparameterizationNormalInverseNormal(tf.keras.layers.Layer):


    _map = {
        'mu': 0,
        'sigma_e': 1,
        'sample': 2,
        'sigma_a': 3
    }
    _n_params = len(_map)
    _recast_map = {
        'mu': 0,
        'sigma_epi': 1,
        'sigma_alea': 2
    }
    _n_recast_params = len(_recast_map)


    def __init__(self, units, **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self._n_outputs = self._n_params * self.units
        self._n_recast_outputs = self._n_recast_params * self.units

        self._fuzz = tf.constant([get_fuzz_factor(self.dtype)], dtype=self.dtype)
        self._epistemic = DenseReparameterizationEpistemic(self.units, name=self.name+'_epistemic')
        self._aleatoric = Dense(self.units, activation='softplus', name=self.name+'_aleatoric')


    # Output: Shape(batch_size, n_outputs)
    @tf.function
    def call(self, inputs):
        epistemic_outputs = self._epistemic(inputs)
        aleatoric_stddevs = self._aleatoric(inputs) + self._fuzz
        return tf.concat([epistemic_outputs, aleatoric_stddevs], axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def recast_to_prediction_epistemic_aleatoric(self, outputs):
        indices = []
        indices.extend([ii for ii in range(self._map['mu'] * self.units, self._map['mu'] * self.units + self.units)])
        indices.extend([ii for ii in range(self._map['sigma_e'] * self.units, self._map['sigma_e'] * self.units + self.units)])
        indices.extend([ii for ii in range(self._map['sigma_a'] * self.units, self._map['sigma_a'] * self.units + self.units)])
        return tf.gather(outputs, indices=indices, axis=-1)


    # Output: Shape(batch_size, n_recast_outputs)
    @tf.function
    def _recast(self, outputs):
        return self.recast_to_prediction_epistemic_aleatoric(outputs)


    def compute_output_shape(self, input_shape):
        return tf.Shape([input_shape[0], self._n_outputs])


    def get_config(self):
        base_config = super().get_config()
        config = {
            'units': self.units,
        }
        return {**base_config, **config}



# ------ LOSSES ------


class NormalNLLLoss(tf.keras.losses.Loss):


    def __init__(self, name='nll', dtype=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype

        self._fuzz = tf.constant([get_fuzz_factor(self.dtype)], dtype=self.dtype)


    @tf.function
    def call(self, target_values, distribution_moments):
        targets, _ = tf.unstack(target_values, axis=-1)
        distribution_locs, distribution_scales = tf.unstack(distribution_moments, axis=-1)
        #distributions = tfd.Normal(loc=distribution_locs, scale=distribution_scales)
        #loss = -distributions.log_prob(targets)
        log_prefactor = tf.math.log(2.0 * np.pi * tf.math.pow(distribution_scales, 2) + self._fuzz)
        log_shape = tf.math.divide_no_nan(tf.math.pow(targets - distribution_locs, 2), tf.math.pow(distribution_scales, 2) + self._fuzz)
        loss = 0.5 * (log_prefactor + log_shape)
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



class NormalNormalKLDivLoss(tf.keras.losses.Loss):


    def __init__(self, name='kld', dtype=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype


    @tf.function
    def call(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = tf.unstack(prior_moments, axis=-1)
        posterior_locs, posterior_scales = tf.unstack(posterior_moments, axis=-1)
        priors = tfd.Normal(loc=prior_locs, scale=prior_scales)
        posteriors = tfd.Normal(loc=posterior_locs, scale=posterior_scales)
        loss = tfd.kl_divergence(priors, posteriors)
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



class NormalNormalFisherRaoLoss(tf.keras.losses.Loss):


    def __init__(self, name='fr', dtype=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype


    @tf.function
    def call(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = tf.unstack(prior_moments, axis=-1)
        posterior_locs, posterior_scales = tf.unstack(posterior_moments, axis=-1)
        distances = tf.math.pow(posterior_locs - prior_locs, 2)
        numerator_scales = tf.math.pow(posterior_scales - prior_scales, 2)
        denominator_scales = tf.math.pow(posterior_scales + prior_scales, 2)
        numerator = distances + 2.0 * numerator_scales
        denominator = distances + 2.0 * denominator_scales
        argument = tf.math.divide_no_nan(tf.math.sqrt(numerator), tf.math.sqrt(denominator))
        loss = tf.math.atanh(argument)
        #loss = -1.0 * tf.math.log(1.0 - argument)
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



class NormalNormalHighUncertaintyLoss(tf.keras.losses.Loss):


    def __init__(self, name='unc', dtype=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype


    @tf.function
    def call(self, prior_moments, posterior_moments):
        prior_locs, prior_scales = tf.unstack(prior_moments, axis=-1)
        posterior_locs, posterior_scales = tf.unstack(posterior_moments, axis=-1)
        #loss = tf.math.sqrt(tf.math.divide_no_nan(tf.math.pow(posterior_scales, 2), tf.math.pow(posterior_locs, 2)))
        #loss = tf.identity(posterior_scales)
        loss = tf.math.pow(tf.math.log(posterior_scales) - tf.math.log(prior_scales), 2)
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



class NoiseContrastivePriorLoss(tf.keras.losses.Loss):


    _possible_distance_losses = [
        'fisher_rao',
        'kl_divergence',
    ]


    def __init__(
        self,
        likelihood_weight=1.0,
        epistemic_weight=1.0,
        aleatoric_weight=1.0,
        distance_loss='fisher_rao',
        name='ncp',
        reduction='sum',
        dtype=None,
        **kwargs
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype

        self._likelihood_weight = likelihood_weight
        self._epistemic_weight = epistemic_weight
        self._aleatoric_weight = aleatoric_weight
        self._distance_loss = distance_loss if distance_loss in self._possible_distance_losses else self._possible_distance_losses[0]
        self._likelihood_loss_fn = NormalNLLLoss(name=self.name+'_nll', reduction=reduction, dtype=self.dtype)
        if self._distance_loss == 'kl_divergence':
            self._epistemic_loss_fn = NormalNormalKLDivLoss(name=self.name+'_epi_kld', reduction=reduction, dtype=self.dtype)
            self._aleatoric_loss_fn = NormalNormalKLDivLoss(name=self.name+'_alea_kld', reduction=reduction, dtype=self.dtype)
        else:  # 'fisher_rao'
            self._epistemic_loss_fn = NormalNormalFisherRaoLoss(name=self.name+'_epi_fr', reduction=reduction, dtype=self.dtype)
            #self._aleatoric_loss_fn = NormalNormalFisherRaoLoss(name=self.name+'_alea_fr', reduction=reduction, dtype=self.dtype)
            self._aleatoric_loss_fn = NormalNormalHighUncertaintyLoss(name=self.name+'_alea_unc', reduction=reduction, dtype=self.dtype)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = tf.constant(self._likelihood_weight, dtype=self.dtype)
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_model_distance_loss(self, targets, predictions):
        weight = tf.constant(self._epistemic_weight, dtype=self.dtype)
        base = self._epistemic_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_noise_distance_loss(self, targets, predictions):
        weight = tf.constant(self._aleatoric_weight, dtype=self.dtype)
        base = self._aleatoric_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments, loss_terms) -> Output: Shape([batch_size])
    @tf.function
    def call(self, targets, predictions):
        target_values, model_prior_moments, noise_prior_moments = tf.unstack(targets, axis=-1)
        prediction_distribution_moments, model_posterior_moments, noise_posterior_moments = tf.unstack(predictions, axis=-1)
        likelihood_loss = self._calculate_likelihood_loss(target_values, prediction_distribution_moments)
        epistemic_loss = self._calculate_model_distance_loss(model_prior_moments, model_posterior_moments)
        aleatoric_loss = self._calculate_noise_distance_loss(noise_prior_moments, noise_posterior_moments)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss


    def get_config(self):
        base_config = super().get_config()
        config = {
            'likelihood_weight': self._likelihood_weight,
            'epistemic_weight': self._epistemic_weight,
            'aleatoric_weight': self._aleatoric_weight,
            'distance_loss': self._distance_loss,
        }
        return {**base_config, **config}



class MultiOutputNoiseContrastivePriorLoss(tf.keras.losses.Loss):


    _possible_distance_losses = [
        'fisher_rao',
        'kl_divergence',
    ]


    def __init__(
        self,
        n_outputs,
        likelihood_weights,
        epistemic_weights,
        aleatoric_weights,
        distance_loss,
        name='multi_ncp',
        reduction='sum',
        dtype=None,
        **kwargs
    ):

        super().__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = dtype if dtype is not None else default_dtype

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._likelihood_weights = []
        self._epistemic_weights = []
        self._aleatoric_weights = []
        self._distance_loss = distance_loss if distance_loss in self._possible_distance_losses else self._possible_distance_losses[0]
        for ii in range(self.n_outputs):
            nll_w = 1.0
            epi_w = 1.0
            alea_w = 1.0
            unc_w = 1.0
            if isinstance(likelihood_weights, (list, tuple)):
                nll_w = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
            if isinstance(epistemic_weights, (list, tuple)):
                epi_w = epistemic_weights[ii] if ii < len(epistemic_weights) else epistemic_weights[-1]
            if isinstance(aleatoric_weights, (list, tuple)):
                alea_w = aleatoric_weights[ii] if ii < len(aleatoric_weights) else aleatoric_weights[-1]
            self._loss_fns[ii] = NoiseContrastivePriorLoss(
                nll_w,
                epi_w,
                alea_w,
                self._distance_loss,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                dtype=self.dtype
            )
            self._likelihood_weights.append(nll_w)
            self._epistemic_weights.append(epi_w)
            self._aleatoric_weights.append(alea_w)


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
    def _calculate_model_distance_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_model_distance_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    @tf.function
    def _calculate_noise_distance_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_noise_distance_loss(target_stack[ii], prediction_stack[ii]))
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
        base_config = super().get_config()
        config = {
            'likelihood_weights': self._likelihood_weights,
            'epistemic_weights': self._epistemic_weights,
            'aleatoric_weights': self._aleatoric_weights,
            'distance_loss': self._distance_loss,
        }
        return {**base_config, **config}


