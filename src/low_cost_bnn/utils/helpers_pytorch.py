from pathlib import Path
import numpy as np
import torch

default_dtype = torch.get_default_dtype()


def create_data_loader(data_tuple, batch_size=None, buffer_size=None, seed=None):
    dataset = torch.utils.data.TensorDataset(*data_tuple)
    shuffle = True if isinstance(buffer_size, int) else None
    generator = None
    if isinstance(seed, int):
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    return loader


def create_scheduled_adam_optimizer(model, learning_rate, decay_steps, decay_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_steps, gamma=decay_rate)
    return optimizer, scheduler


def create_model(n_input, n_output, n_common, common_nodes=None, special_nodes=None, relative_reg=0.1, style='ncp', name=f'ncp', verbosity=0):
    from ..models.pytorch import TrainableUncertaintyAwareNN
    parameterization_layer = torch.nn.Identity
    if style == 'ncp':
        from ..models.noise_contrastive_pytorch import DenseReparameterizationNormalInverseNormal
        parameterization_layer = DenseReparameterizationNormalInverseNormal
    if style == 'evidential':
        from ..models.evidential_pytorch import DenseReparameterizationNormalInverseGamma
        parameterization_layer = DenseReparameterizationNormalInverseGamma
    model = TrainableUncertaintyAwareNN(
        parameterization_layer,
        n_input,
        n_output,
        n_common,
        common_nodes=common_nodes,
        special_nodes=special_nodes,
        relative_reg=relative_reg,
        name=name
    )
    return model


def create_loss_function(n_outputs, style='ncp', verbosity=0, **kwargs):
    if style == 'ncp':
        return create_noise_contrastive_prior_loss_function(n_outputs, verbosity=verbosity, **kwargs)
    elif style == 'evidential':
        return create_evidential_loss_function(n_outputs, verbosity=verbosity, **kwargs)
    else:
        raise KeyError('Invalid loss function style passed to loss function generator.')


def create_noise_contrastive_prior_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, verbosity=0):
    if n_outputs > 1:
        from ..models.noise_contrastive_pytorch import MultiOutputNoiseContrastivePriorLoss
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, reduction='sum')
    elif n_outputs == 1:
        from ..models.noise_contrastive_pytorch import NoiseContrastivePriorLoss
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


def create_evidential_loss_function(n_outputs, nll_weights, evi_weights, verbosity=0):
    if n_outputs > 1:
        from ..models.evidential_pytorch import MultiOutputEvidentialLoss
        return MultiOutputEvidentialLoss(n_outputs, nll_weights, evi_weights, reduction='sum')
    elif n_outputs == 1:
        from ..models.evidential_pytorch import EvidentialLoss
        return EvidentialLoss(nll_weights, evi_weights, reduction='sum')
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


def wrap_model(model, scaler_in, scaler_out):
    from ..models.pytorch import TrainedUncertaintyAwareNN
    try:
        input_mean = scaler_in.mean_
        input_var = scaler_in.var_
        output_mean = scaler_out.mean_
        output_var = scaler_out.var_
        input_tags = scaler_in.feature_names_in_.tolist()
        output_tags = scaler_out.feature_names_in_.tolist()
    except:
        input_mean = np.array([0.0] * model.n_inputs)
        input_var = np.array([1.0] * model.n_inputs)
        output_mean = np.array([0.0] * model.n_outputs)
        output_var = np.array([1.0] * model.n_outputs)
        input_tags = None
        output_tags = None
    wrapper = TrainedUncertaintyAwareNN(
        model,
        input_mean,
        input_var,
        output_mean,
        output_var,
        input_tags,
        output_tags,
        name=f'wrapped_{model.name}'
    )
    return wrapper


def load_model(model_path):
    model = None
    if isinstance(model_path, Path) and model_path.is_file():
        from ..models.pytorch import TrainedUncertaintyAwareNN
        model_save_dict = torch.load(model_path)
        model = TrainedUncertaintyAwareNN.from_config(model_save_dict.get('config_dict', None))
        model.load_state_dict(model_save_dict.get('state_dict', None))
        model.eval()
    else:
        print(f'Specified path, {model_path}, is not a PyTorch custom model file! Aborting!')
    return model


def save_model(model, model_path):
    model_save_dict = {
        'config_dict': model.get_config(),
        'state_dict': model.state_dict()
    }
    torch.save(model_save_dict, model_path)


