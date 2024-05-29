import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Lambda, LeakyReLU
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl



# Required custom class for variational layer
class DenseReparameterizationEpistemic(tfpl.DenseReparameterization):


    def __init__(self, units, **kwargs):

        super(DenseReparameterizationEpistemic, self).__init__(units, **kwargs)


    def _compute_mean_distribution_moments(self, inputs):
        kernel_mean = self.kernel_posterior.mean()
        kernel_stddev = self.kernel_posterior.stddev()
        bias_mean = self.bias_posterior.mean()
        dist_mean = tf.matmul(inputs, kernel_mean) + bias_mean
        dist_var = tf.matmul(inputs ** 2, kernel_stddev ** 2)
        dist_stddev = tf.sqrt(dist_var)
        return dist_mean, dist_stddev


    @tf.function
    def call(self, inputs):
        samples = super(DenseReparameterizationEpistemic, self).call(inputs)
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return means, stddevs, samples


    def get_config(self):
        base_config = super(DenseReparameterizationEpistemic, self).get_config()
        config = {
        }
        return {**base_config, **config}



class DenseReparameterizationNormalInverseNormal(tf.keras.layers.Layer):


    _n_moments = 4


    def __init__(self, units, **kwargs):

        super(DenseReparameterizationNormalInverseNormal, self).__init__(**kwargs)

        self.units = units
        self._epistemic = DenseReparameterizationEpistemic(self.units, name=self.name+'_epistemic')
        self._aleatoric = Dense(self.units, activation='softplus', name=self.name+'_aleatoric')


    # Output: Shape(batch_size, n_moments)
    @tf.function
    def call(self, inputs):
        epistemic_means, epistemic_stddevs, aleatoric_samples = self._epistemic(inputs)
        aleatoric_stddevs = self._aleatoric(inputs)
        return tf.concat([epistemic_means, epistemic_stddevs, aleatoric_samples, aleatoric_stddevs], axis=-1)


    def get_config(self):
        base_config = super(DenseReparameterizationNormalInverseNormal, self).get_config()
        config = {
            'units': self.units,
        }
        return {**base_config, **config}



class TrainableLowCostBNN(tf.keras.models.Model):


    _parameterization_class = DenseReparameterizationNormalInverseNormal


    def __init__(self, n_input, n_output, n_hidden, n_special, **kwargs):

        super(TrainableLowCostBNN, self).__init__(**kwargs)

        self.n_inputs = n_input
        self.n_outputs = n_output
        self.n_hiddens = list(n_hidden) if isinstance(n_hidden, (list, tuple)) else [20]
        self.n_specials = list(n_special) if isinstance(n_special, (list, tuple)) else [self.n_hiddens[0]]
        while len(self.n_specials) < self.n_outputs:
            self.n_specials.append(self.n_specials[-1])
        if len(self.n_specials) > self.n_outputs:
            self.n_specials = self.n_specials[:self.n_outputs]

        self._base_activation = LeakyReLU(alpha=0.2)

        self._common_layers = tf.keras.Sequential()
        for ii in range(len(self.n_hiddens)):
            self._common_layers.add(Dense(self.n_hiddens[ii], activation=self._base_activation, name=f'common{ii}'))

        self._output_channels = [None] * self.n_outputs
        for jj in range(self.n_outputs):
            channel = tf.keras.Sequential()
            channel.add(Dense(self.n_specials[jj], activation=self._base_activation, name=f'specialized{jj}'))
            channel.add(DenseReparameterizationNormalInverseNormal(1, name=f'output{jj}'))
            self._output_channels[jj] = channel

        self.build((None, self.n_inputs))


    # Output: Shape(batch_size, n_moments, n_outputs)
    @tf.function
    def call(self, inputs):
        commons = self._common_layers(inputs)
        specials = []
        for jj in range(len(self._output_channels)):
            specials.append(self._output_channels[jj](commons))
        outputs = tf.stack(specials, axis=-1)
        return outputs


    def get_config(self):
        base_config = super(TrainableLowCostBNN, self).get_config()
        config = {
            'n_input': self.n_inputs,
            'n_output': self.n_outputs,
            'n_hidden': self.n_hiddens,
            'n_special': self.n_specials,
        }
        return {**base_config, **config}



class TrainedLowCostBNN(tf.keras.models.Model):

    
    def __init__(self, trained_model, input_mean, input_var, output_mean, output_var, input_tags=None, output_tags=None, **kwargs):

        super(TrainedLowCostBNN, self).__init__(**kwargs)

        self._input_mean = input_mean
        self._input_variance = input_var
        self._output_mean = output_mean
        self._output_variance = output_var
        self._input_tags = input_tags
        self._output_tags = output_tags

        self.n_inputs = len(self._input_mean)
        self.n_outputs = len(self._output_mean)

        extended_output_mean = []
        for ii in range(self.n_outputs):
            temp = [self._output_mean[ii], 0.0, self._output_mean[ii], 0.0]
            extended_output_mean.extend(temp)
        output_mean = tf.constant(extended_output_mean, dtype=tf.keras.backend.floatx())
        extended_output_variance = []
        for ii in range(self.n_outputs):
            temp = [self._output_variance[ii], self._output_variance[ii], self._output_variance[ii], self._output_variance[ii]]
            extended_output_variance.extend(temp)
        output_variance = tf.constant(extended_output_variance, dtype=tf.keras.backend.floatx())
        self._extended_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags):
                temp = [self._output_tags[ii]+'_mu', self._output_tags[ii]+'_epi_sigma', self._output_tags[ii]+'_mu_sample', self._output_tags[ii]+'_alea_sigma']
                self._extended_output_tags.extend(temp)

        self._input_norm = tf.keras.layers.Normalization(axis=-1, mean=self._input_mean, variance=self._input_variance)
        self._trained_model = trained_model
        self._output_denorm = tf.keras.layers.Normalization(axis=-1, mean=output_mean, variance=output_variance, invert=True)

        self.build((None, self.n_inputs))


    @property
    def get_model(self):
        return self._trained_model


    # Output: Shape(batch_size, n_moments * n_outputs)
    @tf.function
    def call(self, inputs):
        n_moments = self._trained_model._parameterization_class._n_moments
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        shaped_outputs = tf.reshape(norm_outputs, shape=[-1, self.n_outputs * n_moments])
        outputs = self._output_denorm(shaped_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=tf.keras.backend.floatx())
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._expanded_output_tags, dtype=input_df.dtypes.iloc[0])
        drop_tags = [tag for tag in self._extended_output_tags if tag.endswith('_sample')]
        return output_df.drop(drop_tags, axis=1)


    def get_config(self):
        base_config = super(TrainedLowCostBNN, self).get_config()
        config = {
            'input_mean': self._input_mean,
            'input_var': self._input_variance,
            'output_mean': self._output_mean,
            'output_var': self._output_variance,
            'input_tags': self._input_tags,
            'output_tags': self._output_tags,
        }
        return {**base_config, **config}



class DistributionNLLLoss(tf.keras.losses.Loss):


    def __init__(self, name='nll', **kwargs):

        super(DistributionNLLLoss, self).__init__(name=name, **kwargs)


    @tf.function
    def call(self, target_values, distribution_moments):
        distributions = tfd.Normal(loc=distribution_moments[..., 0], scale=distribution_moments[..., 1])
        loss = -distributions.log_prob(target_values[..., 0])
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss)
        return loss


    def get_config(self):
        base_config = super(EpistemicLoss, self).get_config()
        config = {
        }
        return {**base_config, **config}



class DistributionKLDivLoss(tf.keras.losses.Loss):


    def __init__(self, name='kld', **kwargs):

        super(DistributionKLDivLoss, self).__init__(name=name, **kwargs)


    @tf.function
    def call(self, prior_moments, posterior_moments):
        priors = tfd.Normal(loc=prior_moments[..., 0], scale=prior_moments[..., 1])
        posteriors = tfd.Normal(loc=posterior_moments[..., 0], scale=posterior_moments[..., 1])
        loss = tfd.kl_divergence(priors, posteriors)
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss)
        return loss


    def get_config(self):
        base_config = super(EpistemicLoss, self).get_config()
        config = {
        }
        return {**base_config, **config}



class NoiseContrastivePriorLoss(tf.keras.losses.Loss):


    def __init__(self, likelihood_weight=1.0, epistemic_weight=1.0, aleatoric_weight=1.0, name='ncp', reduction='sum', **kwargs):

        super(NoiseContrastivePriorLoss, self).__init__(name=name, reduction=reduction, **kwargs)

        self.dtype = tf.keras.backend.floatx()
        self._likelihood_weight = likelihood_weight
        self._epistemic_weight = epistemic_weight
        self._aleatoric_weight = aleatoric_weight
        self._likelihood_loss_fn = DistributionNLLLoss(name=self.name+'_nll', reduction=reduction, **kwargs)
        self._epistemic_loss_fn = DistributionKLDivLoss(name=self.name+'_epi_kld', reduction=reduction, **kwargs)
        self._aleatoric_loss_fn = DistributionKLDivLoss(name=self.name+'_alea_kld', reduction=reduction, **kwargs)


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_likelihood_loss(self, targets, predictions):
        weight = tf.constant(self._likelihood_weight, dtype=self.dtype)
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_model_divergence_loss(self, targets, predictions):
        weight = tf.constant(self._epistemic_weight, dtype=self.dtype)
        base = self._epistemic_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    # Input: Shape(batch_size, dist_moments) -> Output: Shape([batch_size])
    @tf.function
    def _calculate_noise_divergence_loss(self, targets, predictions):
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
        epistemic_loss = self._calculate_model_divergence_loss(model_prior_moments, model_posterior_moments)
        aleatoric_loss = self._calculate_noise_divergence_loss(noise_prior_moments, noise_posterior_moments)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss


    def get_config(self):
        base_config = super(EpistemicLoss, self).get_config()
        config = {
            'likelihood_weight': self._likelihood_weight,
            'epistemic_weight': self._epistemic_weight,
            'aleatoric_weight': self._aleatoric_weight,
        }
        return {**base_config, **config}



class MultiOutputNoiseContrastivePriorLoss(tf.keras.losses.Loss):


    def __init__(self, n_outputs, likelihood_weights, epistemic_weights, aleatoric_weights, name='multi_ncp', reduction='sum', **kwargs):

        super(MultiOutputNoiseContrastivePriorLoss, self).__init__(name=name, reduction=reduction, **kwargs)

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._likelihood_weights = []
        self._epistemic_weights = []
        self._aleatoric_weights = []
        for ii in range(self.n_outputs):
            nll_w = 1.0
            epi_w = 1.0
            alea_w = 1.0
            if isinstance(likelihood_weights, (list, tuple)):
                nll_w = likelihood_weights[ii] if ii < len(likelihood_weights) else likelihood_weights[-1]
            if isinstance(epistemic_weights, (list, tuple)):
                epi_w = epistemic_weights[ii] if ii < len(epistemic_weights) else epistemic_weights[-1]
            if isinstance(aleatoric_weights, (list, tuple)):
                alea_w = aleatoric_weights[ii] if ii < len(aleatoric_weights) else aleatoric_weights[-1]
            self._loss_fns[ii] = NoiseContrastivePriorLoss(nll_w, epi_w, alea_w, name=f'{self.name}_out{ii}', reduction=self.reduction)
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
    def _calculate_model_divergence_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_model_divergence_loss(target_stack[ii], prediction_stack[ii]))
        return tf.stack(losses, axis=-1)


    # Input: Shape(batch_size, dist_moments, n_outputs) -> Output: Shape([batch_size], n_outputs)
    @tf.function
    def _calculate_noise_divergence_loss(self, targets, predictions):
        target_stack = tf.unstack(targets, axis=-1)
        prediction_stack = tf.unstack(predictions, axis=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_noise_divergence_loss(target_stack[ii], prediction_stack[ii]))
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
        base_config = super(MultiOutputNoiseContrastivePriorLoss, self).get_config()
        config = {
            'likelihood_weights': self._likelihood_weights,
            'epistemic_weights': self._epistemic_weights,
            'aleatoric_weights': self._aleatoric_weights
        }
        return {**base_config, **config}


