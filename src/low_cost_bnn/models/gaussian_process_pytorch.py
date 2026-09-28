import math
import torch
from torch.nn import Parameter, Linear
from ..utils.helpers_pytorch import default_dtype, default_device


# Mean-field approximation factor for the logistic-Gaussian integral, lambda = pi / 8 (Lu et al., 2020)
default_mean_field_factor = math.pi / 8.0



# ------ LAYERS ------


# Mirrors the tf.keras RandomFourierFeatures layer, which has no equivalent in PyTorch
class RandomFourierFeatures(torch.nn.Module):


    _SUPPORTED_KERNEL_TYPES = ('gaussian', 'laplacian')


    def __init__(
        self,
        in_features,
        out_features,
        kernel_initializer='gaussian',
        scale=None,
        trainable=False,
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        if kernel_initializer not in self._SUPPORTED_KERNEL_TYPES:
            raise ValueError(f'"kernel_initializer" must be one of {self._SUPPORTED_KERNEL_TYPES}, got {kernel_initializer}.')

        super().__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_initializer = kernel_initializer
        self.trainable = trainable

        # Default scale taken from tf.keras implementation
        if scale is None:
            scale = math.sqrt(self.in_features / 2.0) if self.kernel_initializer == 'gaussian' else 1.0

        # Random features are fixed, only the kernel scale can be trained
        unscaled_kernel = torch.empty((self.in_features, self.out_features), **self.factory_kwargs)
        if self.kernel_initializer == 'gaussian':
            torch.nn.init.normal_(unscaled_kernel, mean=0.0, std=1.0)
        else:
            unscaled_kernel.cauchy_(median=0.0, sigma=1.0)
        self.register_buffer('unscaled_kernel', unscaled_kernel)
        bias = torch.empty(self.out_features, **self.factory_kwargs)
        torch.nn.init.uniform_(bias, a=0.0, b=2.0 * math.pi)
        self.register_buffer('bias', bias)
        self.kernel_scale = Parameter(torch.tensor([scale], **self.factory_kwargs), requires_grad=self.trainable)


    # Output: Shape(batch_size, out_features)
    def forward(self, inputs):
        kernel = self.unscaled_kernel / self.kernel_scale
        outputs = torch.matmul(inputs, kernel) + self.bias
        return math.sqrt(2.0 / float(self.out_features)) * torch.cos(outputs)



# Taken from tf-models-official and modified
class RandomFeatureGaussianProcess(torch.nn.Module):


    def __init__(
        self,
        in_features,
        out_features,
        num_inducing=1024,
        gp_kernel_type='gaussian',
        gp_kernel_scale=1.0,
        gp_output_bias=0.0,
        gp_kernel_scale_trainable=False,
        gp_output_bias_trainable=False,
        gp_cov_momentum=-1,
        gp_cov_ridge_penalty=1.0,
        gp_cov_diagonal=False,
        scale_random_features=True,
        use_custom_random_features=True,
        custom_random_features_initializer=None,
        custom_random_features_activation=None,
        l2_regularization=0.0,
        gp_cov_likelihood='binary_logistic',
        return_gp_cov=True,
        return_random_features=False,
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.num_inducing = num_inducing

        self.gp_input_scale = 1.0 / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2.0 / float(num_inducing))

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

        # Stored for configuration parity only, the TensorFlow training step does not apply it either
        self.l2_regularization = l2_regularization

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_likelihood = gp_cov_likelihood
        self.gp_cov_diagonal = gp_cov_diagonal

        if self.use_custom_random_features:
            # Default to Gaussian RBF kernel
            self.random_features_bias_initializer = lambda tensor: torch.nn.init.uniform_(tensor, a=0.0, b=2.0 * math.pi)
            if self.custom_random_features_initializer is None:
                self.custom_random_features_initializer = lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
            if self.custom_random_features_activation is None:
                self.custom_random_features_activation = torch.cos

        self.build()


    def build(self):

        self._random_feature = self._make_random_feature_layer()

        if self.return_gp_cov:
            self._gp_cov_layer = LaplaceRandomFeatureCovariance(
                self.num_inducing,
                momentum=self.gp_cov_momentum,
                ridge_penalty=self.gp_cov_ridge_penalty,
                likelihood=self.gp_cov_likelihood,
                diagonal=self.gp_cov_diagonal,
                **self.factory_kwargs
            )

        self._gp_output_layer = Linear(self.num_inducing, self.out_features, bias=False, **self.factory_kwargs)

        self._gp_output_bias = Parameter(
            torch.tensor([self.gp_output_bias] * self.out_features, **self.factory_kwargs),
            requires_grad=self.gp_output_bias_trainable
        )


    def _make_random_feature_layer(self):
        if self.use_custom_random_features:
            layer = Linear(self.in_features, self.num_inducing, bias=True, **self.factory_kwargs)
            with torch.no_grad():
                self.custom_random_features_initializer(layer.weight)
                self.random_features_bias_initializer(layer.bias)
            return layer
        else:
            return RandomFourierFeatures(
                self.in_features,
                self.num_inducing,
                kernel_initializer=self.gp_kernel_type,
                scale=self.gp_kernel_scale,
                trainable=self.gp_kernel_scale_trainable,
                **self.factory_kwargs
            )


    def reset_covariance_matrix(self):
        # Required at the beginning of every epoch!
        self._gp_cov_layer.reset_precision_matrix()


    # Output: Shape(batch_size, num_inducing)
    def compute_random_features(self, inputs):
        gp_inputs = inputs * self.gp_input_scale
        gp_feature = self._random_feature(gp_inputs)
        if self.use_custom_random_features:
            gp_feature = self.custom_random_features_activation(gp_feature)
        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_feature_scale
        return gp_feature


    # Output: Shape(batch_size, batch_size)
    def get_covariance(self, inputs):
        gp_feature = self.compute_random_features(inputs)
        return self._gp_cov_layer.compute_predictive_covariance(gp_feature)


    # Output: [Shape(batch_size, out_features), Shape(batch_size, batch_size) or Shape(batch_size) if diagonal, Shape(batch_size, num_inducing)]
    def forward(self, inputs):

        gp_feature = self.compute_random_features(inputs)

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if self.return_gp_cov:
            gp_covmat = self._gp_cov_layer(gp_feature, gp_output)

        # Assembles model output.
        model_output = [gp_output]
        if self.return_gp_cov:
            model_output.append(gp_covmat)
        if self.return_random_features:
            model_output.append(gp_feature)

        return model_output



# Taken from tf-models-official and modified
class LaplaceRandomFeatureCovariance(torch.nn.Module):


    _SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')


    def __init__(
        self,
        in_features,
        momentum=0.999,
        ridge_penalty=1.,
        likelihood='binary_logistic',
        diagonal=False,
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        if likelihood not in self._SUPPORTED_LIKELIHOOD:
            raise ValueError(f'"likelihood" must be one of {self._SUPPORTED_LIKELIHOOD}, got {likelihood}.')

        super().__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.likelihood = likelihood
        self.diagonal = diagonal

        self.build()


    def build(self):

        gp_feature_dim = self.in_features

        # Posterior precision matrix for the GP's random feature coefficients, stored as buffers to be saved with the model
        self.register_buffer('initial_precision_matrix', self.ridge_penalty * torch.eye(gp_feature_dim, **self.factory_kwargs))
        self.register_buffer('precision_matrix', self.initial_precision_matrix.clone())


    @torch.no_grad()
    def make_precision_matrix_update_op(self, gp_feature, logits, precision_matrix):

        gp_feature = gp_feature.detach()
        batch_size = float(gp_feature.shape[0])

        # Computes batch-specific normalized precision matrix
        if self.likelihood == 'binary_logistic':
            prob = torch.sigmoid(logits.detach())
            prob_multiplier = prob * (1.0 - prob)
            if logits.shape[-1] > 1:
                prob_multiplier = torch.amax(prob_multiplier, dim=-1, keepdim=True)
        elif self.likelihood == 'poisson':
            prob_multiplier = torch.exp(logits.detach())
            if logits.shape[-1] > 1:
                prob_multiplier = torch.amax(prob_multiplier, dim=-1, keepdim=True)
        else:
            prob_multiplier = torch.ones(gp_feature.shape, **self.factory_kwargs)

        gp_feature_adjusted = torch.sqrt(prob_multiplier) * gp_feature
        precision_matrix_minibatch = torch.matmul(torch.transpose(gp_feature_adjusted, 0, 1), gp_feature_adjusted)

        # Updates the population-wise precision matrix
        if self.momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision matrices
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.momentum * precision_matrix + (1. - self.momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum
            # Only pass data through once if using this option
            precision_matrix_new = precision_matrix + precision_matrix_minibatch

        return precision_matrix.copy_(precision_matrix_new)


    @torch.no_grad()
    def reset_precision_matrix(self):
        self.precision_matrix.copy_(self.initial_precision_matrix)


    # Output: Shape(batch_size, batch_size)
    def compute_predictive_covariance(self, gp_feature):

        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, torch.transpose(gp_feature, 0, 1)) * self.ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)

        return gp_cov_matrix


    # Output: Shape(batch_size), equal to the diagonal of compute_predictive_covariance without building the full matrix
    def compute_predictive_variance(self, gp_feature):

        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the variance of the gp prediction, each entry only depends on its own features
        cov_feature_product = torch.matmul(gp_feature, feature_cov_matrix) * self.ridge_penalty
        gp_variance = torch.sum(cov_feature_product * gp_feature, dim=-1)

        return gp_variance


    # Output: Shape(batch_size, batch_size), or Shape(batch_size) if diagonal
    def forward(self, inputs, logits=None):

        batch_size = inputs.shape[0]

        if self.training:
            self.make_precision_matrix_update_op(
                gp_feature=inputs,
                logits=logits,
                precision_matrix=self.precision_matrix
            )
            # Return null during training
            if self.diagonal:
                return torch.ones((batch_size, ), **self.factory_kwargs)
            return torch.eye(batch_size, **self.factory_kwargs)
        else:
            # Return covariance estimate during prediction
            if self.diagonal:
                return self.compute_predictive_variance(gp_feature=inputs)
            return self.compute_predictive_covariance(gp_feature=inputs)



class DenseReparameterizationGaussianProcess(torch.nn.Module):


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


    def __init__(
        self,
        in_features,
        out_features,
        threshold=0.5,
        mean_field_factor=default_mean_field_factor,
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self._n_outputs = self._n_params * self.out_features
        self._n_recast_outputs = self._n_recast_params * self.out_features
        self._threshold = float(threshold) if isinstance(threshold, float) else 0.5
        self._mean_field_factor = float(mean_field_factor)

        # Internal random features return cos(W * h + B), scale_random_features multiplies by sqrt(2 / D)
        # Internal Linear layer acts as the trainable beta vector
        # Diagonal covariance is equivalent to the diagonal of the full batch covariance matrix, but scales linearly with batch size
        self._gaussian_layer = RandomFeatureGaussianProcess(
            self.in_features,
            self.out_features,
            num_inducing=1024,
            gp_cov_momentum=-1,
            gp_cov_ridge_penalty=1.0,
            gp_cov_diagonal=True,
            scale_random_features=True,
            l2_regularization=0.0,
            gp_cov_likelihood='binary_logistic',
            return_gp_cov=True,
            return_random_features=False,
            **self.factory_kwargs
        )


    # Output: Shape(batch_size, n_outputs)
    def forward(self, inputs):
        logits, variance = self._gaussian_layer(inputs)
        variance = torch.unsqueeze(variance, dim=-1).expand(logits.shape)
        return torch.cat([logits, variance], dim=-1)


    # Output: Shape(batch_size, batch_size)
    def get_covariance(self, inputs):
        return self._gaussian_layer.get_covariance(inputs)


    # Output: Shape(batch_size, n_recast_outputs)
    def recast_to_prediction_epistemic(self, outputs):
        logit_indices = [ii for ii in range(self._map['logits'] * self.out_features, self._map['logits'] * self.out_features + self.out_features)]
        variance_indices = [ii for ii in range(self._map['variance'] * self.out_features, self._map['variance'] * self.out_features + self.out_features)]
        logits = torch.index_select(outputs, dim=-1, index=torch.tensor(logit_indices, device=outputs.device))
        variance = torch.index_select(outputs, dim=-1, index=torch.tensor(variance_indices, device=outputs.device))
        mean_field_logits = torch.div(logits, torch.sqrt(1.0 + self._mean_field_factor * variance))
        if self.out_features > 1:
            full_probabilities = torch.softmax(mean_field_logits, dim=-1)
            probabilities, maximum_index = torch.max(full_probabilities, dim=-1)
            prediction = maximum_index.to(outputs.dtype)
        else:
            probabilities = torch.squeeze(torch.sigmoid(mean_field_logits), dim=-1)
            threshold_shift = 0.5 - self._threshold
            prediction = torch.round(probabilities + threshold_shift)
        uncertainty = 1.0 - torch.abs(2.0 * probabilities - 1.0)
        return torch.stack([prediction, uncertainty], dim=-1)


    # Output: Shape(batch_size, n_recast_outputs)
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


    def to(self, *args, **kwargs):
        other = super().to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if 'dtype' in other.factory_kwargs and dtype is not None:
            other.factory_kwargs['dtype'] = dtype
        if 'device' in other.factory_kwargs and device is not None:
            other.factory_kwargs['device'] = 'cuda' if 'cuda' in str(device) else 'cpu'
        return other



# ------ LOSSES ------


class CrossEntropyLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        entropy_weight=1.0,
        name='crossentropy',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._entropy_weight = entropy_weight
        self._entropy_loss_fn = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)


    # Input: Shape(batch_size) -> Output: Shape([batch_size])
    def _calculate_entropy_loss(self, targets, predictions):
        weight = torch.tensor([self._entropy_weight], **self.factory_kwargs)
        base = self._entropy_loss_fn(predictions, targets)
        loss = weight * base
        return loss


    def forward(self, targets, predictions):
        entropy_loss = self._calculate_entropy_loss(targets, predictions)
        total_loss = entropy_loss
        return total_loss



class MultiClassCrossEntropyLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        entropy_weight=1.0,
        name='crossentropy',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self._entropy_weight = entropy_weight
        self._entropy_loss_fn = torch.nn.CrossEntropyLoss(reduction=self.reduction)


    # Input: Shape(batch_size), Shape(batch_size, n_classes) -> Output: Shape([batch_size])
    def _calculate_entropy_loss(self, targets, predictions):
        weight = torch.tensor([self._entropy_weight], **self.factory_kwargs)
        base = self._entropy_loss_fn(predictions, targets.long())
        loss = weight * base
        return loss


    def forward(self, targets, predictions):
        entropy_loss = self._calculate_entropy_loss(targets, predictions)
        total_loss = entropy_loss
        return total_loss



class MultiOutputCrossEntropyLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        n_outputs,
        entropy_weights,
        name='multi_crossentropy',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._entropy_weights = [0.0] * self.n_outputs
        for ii in range(self.n_outputs):
            ent_w = 1.0
            if isinstance(entropy_weights, (list, tuple)):
                ent_w = entropy_weights[ii] if ii < len(entropy_weights) else entropy_weights[-1]
            elif isinstance(entropy_weights, (float, int)):
                ent_w = float(entropy_weights)
            self._loss_fns[ii] = CrossEntropyLoss(
                ent_w,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                **self.factory_kwargs
            )
            self._entropy_weights[ii] = ent_w


    # Input: Shape(batch_size, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_entropy_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_entropy_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    def forward(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii](target_stack[ii], prediction_stack[ii]))
        total_loss = torch.stack(losses, dim=-1)
        if self.reduction == 'mean':
            total_loss = torch.mean(total_loss)
        elif self.reduction == 'sum':
            total_loss = torch.sum(total_loss)
        return total_loss



class MultiOutputMultiClassCrossEntropyLoss(torch.nn.modules.loss._Loss):


    def __init__(
        self,
        n_outputs,
        entropy_weights,
        name='multi_crossentropy',
        reduction='sum',
        dtype=default_dtype,
        device=default_device,
        **kwargs
    ):

        super().__init__(reduction=reduction, **kwargs)

        self.name = name
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_outputs = n_outputs
        self._loss_fns = [None] * self.n_outputs
        self._entropy_weights = [0.0] * self.n_outputs
        for ii in range(self.n_outputs):
            ent_w = 1.0
            if isinstance(entropy_weights, (list, tuple)):
                ent_w = entropy_weights[ii] if ii < len(entropy_weights) else entropy_weights[-1]
            elif isinstance(entropy_weights, (float, int)):
                ent_w = float(entropy_weights)
            self._loss_fns[ii] = MultiClassCrossEntropyLoss(
                ent_w,
                name=f'{self.name}_out{ii}',
                reduction=self.reduction,
                **self.factory_kwargs
            )
            self._entropy_weights[ii] = ent_w


    # Input: Shape(batch_size, n_outputs), Shape(batch_size, n_classes, n_outputs) -> Output: Shape([batch_size], n_outputs)
    def _calculate_entropy_loss(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii]._calculate_entropy_loss(target_stack[ii], prediction_stack[ii]))
        return torch.stack(losses, dim=-1)


    def forward(self, targets, predictions):
        target_stack = torch.unbind(targets, dim=-1)
        prediction_stack = torch.unbind(predictions, dim=-1)
        losses = []
        for ii in range(self.n_outputs):
            losses.append(self._loss_fns[ii](target_stack[ii], prediction_stack[ii]))
        total_loss = torch.stack(losses, dim=-1)
        if self.reduction == 'mean':
            total_loss = torch.mean(total_loss)
        elif self.reduction == 'sum':
            total_loss = torch.sum(total_loss)
        return total_loss
