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


    def call(self, inputs):
        samples = super(DenseReparameterizationEpistemic, self).call(inputs)
        means, stddevs = self._compute_mean_distribution_moments(inputs)
        return means, stddevs, samples



class DenseReparameterizationNormalInverseNormal(tf.keras.layers.Layer):


    def __init__(self, units, **kwargs):

        super(DenseReparameterizationNormalInverseNormal, self).__init__(**kwargs)

        self.epistemic = DenseReparameterizationEpistemic(units, name=self.name+'_epistemic')
        self.aleatoric = Dense(units, activation='softplus', name=self.name+'_aleatoric')


    def call(self, inputs):
        epistemic_means, epistemic_stddevs, aleatoric_samples = self.epistemic(inputs)
        aleatoric_stddevs = self.aleatoric(inputs)
        return tf.concat([epistemic_means, epistemic_stddevs, aleatoric_samples, aleatoric_stddevs], axis=-1)



class TrainableLowCostBNN(tf.keras.models.Model):


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


    def call(self, inputs):
        commons = self._common_layers(inputs)
        specials = []
        for jj in range(len(self._output_channels)):
            specials.append(self._output_channels[jj](commons))
        outputs = Concatenate(axis=-1)(specials)
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

        expanded_output_mean = np.array([])
        for ii in range(self.n_outputs):
            temp = np.array([self._output_mean[ii], 0.0, self._output_mean[ii], 0.0])
            expanded_output_mean = np.hstack((expanded_output_mean, temp))
        expanded_output_variance = np.array([])
        for ii in range(self.n_outputs):
            temp = np.array([self._output_variance[ii], self._output_variance[ii], self._output_variance[ii], self._output_variance[ii]])
            expanded_output_variance = np.hstack((expanded_output_variance, temp))
        self._expanded_output_tags = []
        for ii in range(self.n_outputs):
            if isinstance(self._output_tags, (list, tuple)) and ii < len(self._output_tags):
                temp = [self._output_tags[ii]+'_mu', self._output_tags[ii]+'_epi_sigma', self._output_tags[ii]+'_mu_sample', self._output_tags[ii]+'_alea_sigma']
                self._expanded_output_tags.extend(temp)
        self._drop_tags = [tag for tag in self._expanded_output_tags if tag.endswith('_sample')]

        self._input_norm = tf.keras.layers.Normalization(axis=-1, mean=self._input_mean, variance=self._input_variance)
        self._trained_model = trained_model
        self._output_denorm = tf.keras.layers.Normalization(axis=-1, mean=expanded_output_mean, variance=expanded_output_variance, invert=True)

        self.build((None, self.n_inputs))


    @property
    def get_model(self):
        return self._trained_model


    def call(self, inputs):
        norm_inputs = self._input_norm(inputs)
        norm_outputs = self._trained_model(norm_inputs)
        outputs = self._output_denorm(norm_outputs)
        return outputs


    def predict(self, input_df):
        if not isinstance(self._input_tags, (list, tuple)):
            raise ValueError(f'Invalid input column tags provided to {self.__class__.__name__} constructor.')
        if not isinstance(self._output_tags, (list, tuple)):
            raise ValueError(f'Invalid output column tags not provided to {self.__class__.__name__} constructor.')
        inputs = input_df.loc[:, self._input_tags].to_numpy(dtype=tf.keras.backend.floatx())
        outputs = self(inputs)
        output_df = pd.DataFrame(data=outputs, columns=self._expanded_output_tags, dtype=input_df.dtypes.iloc[0]).drop(self._drop_tags, axis=1)
        return output_df


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


    def call(self, targets, distributions):
        loss = -distributions.log_prob(targets)
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


    def call(self, priors, posteriors):
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
        self._likelihood_loss_fn = DistributionNLLLoss(name=self.name+'_nll', reduction=reduction)
        self._epistemic_loss_fn = DistributionKLDivLoss(name=self.name+'_epi_kld', reduction=reduction)
        self._aleatoric_loss_fn = DistributionKLDivLoss(name=self.name+'_alea_kld', reduction=reduction)


    def _calculate_likelihood_loss(self, targets, predictions):
        weight = tf.constant(self._likelihood_weight, dtype=self.dtype)
        base = self._likelihood_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    def _calculate_model_divergence_loss(self, targets, predictions):
        weight = tf.constant(self._epistemic_weight, dtype=self.dtype)
        base = self._epistemic_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    def _calculate_noise_divergence_loss(self, targets, predictions):
        weight = tf.constant(self._aleatoric_weight, dtype=self.dtype)
        base = self._aleatoric_loss_fn(targets, predictions)
        loss = weight * base
        return loss


    def call(self, targets, prediction_dists, model_priors, model_posteriors, noise_priors, noise_posteriors):
        likelihood_loss = self._calculate_likelihood_loss(targets, prediction_dists)
        epistemic_loss = self._calculate_model_divergence_loss(model_priors, model_posteriors)
        aleatoric_loss = self._calculate_noise_divergence_loss(noise_priors, noise_posteriors)
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


    def __init__(self, n_outputs, likelihood_weights, epistemic_weights, aleatoric_weights, name='ncp', reduction='sum', **kwargs):

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
            self._loss_fns[ii] = NoiseContrastivePriorLoss(nll_w, epi_w, alea_w, name=f'nll{ii}', reduction=self.reduction)
            self._likelihood_weights.append(nll_w)
            self._epistemic_weights.append(epi_w)
            self._aleatoric_weights.append(alea_w)


    def _calculate_likelihood_loss(self, targets, predictions):
        losses = [np.nan] * self.n_outputs
        for ii in range(self.n_outputs):
            losses[ii] = self._loss_fns[ii]._calculate_likelihood_loss(targets[ii], predictions[ii])
        return tf.stack(losses, axis=-1)


    def _calculate_model_divergence_loss(self, targets, predictions):
        losses = [np.nan] * self.n_outputs
        for ii in range(self.n_outputs):
            losses[ii] = self._loss_fns[ii]._calculate_model_divergence_loss(targets[ii], predictions[ii])
        return tf.stack(losses, axis=-1)


    def _calculate_noise_divergence_loss(self, targets, predictions):
        losses = [np.nan] * self.n_outputs
        for ii in range(self.n_outputs):
            losses[ii] = self._loss_fns[ii]._calculate_noise_divergence_loss(targets[ii], predictions[ii])
        return tf.stack(losses, axis=-1)


    def call(self, targets, prediction_dists, model_priors, model_posteriors, noise_priors, noise_posteriors):
        likelihood_loss = self._calculate_likelihood_loss(targets, prediction_dists)
        epistemic_loss = self._calculate_model_divergence_loss(model_priors, model_posteriors)
        aleatoric_loss = self._calculate_noise_divergence_loss(noise_priors, noise_posteriors)
        total_loss = likelihood_loss + epistemic_loss + aleatoric_loss
        return total_loss


    def get_config(self):
        base_config = super(MultiOutputNoiseContrastivePriorLoss, self).get_config()
        config = {
            'likelihood_weights': self._likelihood_weights,
            'epistemic_weights': self._epistemic_weights,
            'aleatoric_weights': self._aleatoric_weights
        }
        return {**base_config, **config}


