from pathlib import Path
import numpy as np
import torch

from .helpers import numpy_default_dtype

torch.set_default_dtype(torch.float64 if numpy_default_dtype == np.float64 else torch.float32)
default_dtype = torch.get_default_dtype()
small_eps = torch.tensor([np.finfo(numpy_default_dtype).eps], dtype=default_dtype)


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


def create_noise_contrastive_prior_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, distance_loss, verbosity=0):
    if n_outputs > 1:
        from ..models.noise_contrastive_pytorch import MultiOutputNoiseContrastivePriorLoss
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, distance_loss, reduction='sum')
    elif n_outputs == 1:
        from ..models.noise_contrastive_pytorch import NoiseContrastivePriorLoss
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, distance_loss, reduction='sum')
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


def create_regressor_model(
    n_input,
    n_output,
    n_common,
    common_nodes=None,
    special_nodes=None,
    regpar_l1=0.0,
    regpar_l2=0.0,
    relative_regpar=1.0,
    style='ncp',
    name=f'ncp',
    verbosity=0
):
    from ..models.pytorch import TrainableUncertaintyAwareRegressorNN
    parameterization_layer = torch.nn.Identity
    if style == 'ncp':
        from ..models.noise_contrastive_pytorch import DenseReparameterizationNormalInverseNormal
        parameterization_layer = DenseReparameterizationNormalInverseNormal
    if style == 'evidential':
        from ..models.evidential_pytorch import DenseReparameterizationNormalInverseGamma
        parameterization_layer = DenseReparameterizationNormalInverseGamma
    model = TrainableUncertaintyAwareRegressorNN(
        parameterization_layer,
        n_input,
        n_output,
        n_common,
        common_nodes=common_nodes,
        special_nodes=special_nodes,
        regpar_l1=regpar_l1,
        regpar_l2=regpar_l2,
        relative_regpar=relative_regpar,
        name=name
    )
    return model


def create_regressor_loss_function(n_outputs, style='ncp', verbosity=0, **kwargs):
    if style == 'ncp':
        return create_noise_contrastive_prior_loss_function(n_outputs, verbosity=verbosity, **kwargs)
    elif style == 'evidential':
        return create_evidential_loss_function(n_outputs, verbosity=verbosity, **kwargs)
    else:
        raise KeyError('Invalid loss function style passed to loss function generator.')


def wrap_regressor_model(model, scaler_in, scaler_out):
    from ..models.pytorch import TrainedUncertaintyAwareRegressorNN
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
    wrapper = TrainedUncertaintyAwareRegressorNN(
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


def create_classifier_model():
    return None


def create_classifier_loss_function():
    return None


def wrap_classifier_model(model, scaler_in, names_out):
    from ..models.pytorch import TrainedUncertaintyAwareClassifierNN
    try:
        input_mean = scaler_in.mean_
        input_var = scaler_in.var_
        input_tags = scaler_in.feature_names_in_.tolist()
        output_tags = names_out
    except:
        input_mean = np.array([0.0] * model.n_inputs)
        input_var = np.array([1.0] * model.n_inputs)
        input_tags = None
        output_tags = None
    wrapper = TrainedUncertaintyAwareClassifierNN(
        model,
        input_mean,
        input_var,
        input_tags,
        output_tags,
        name=f'wrapped_{model.name}'
    )
    return wrapper


def load_model(model_path):
    model = None
    if isinstance(model_path, Path) and model_path.is_file():
        model_save_dict = torch.load(model_path)
        config_dict = model_save_dict.get('config_dict', None)
        state_dict = model_save_dict.get('state_dict', None)
        if config_dict.get('class_name', 'TrainedUncertaintyAwareRegressorNN') == 'TrainedUncertaintyAwareRegressorNN':
            from ..models.pytorch import TrainedUncertaintyAwareRegressorNN
            model = TrainedUncertaintyAwareRegressorNN.from_config(config_dict)
            model.load_state_dict(state_dict)
            model.eval()
        elif config_dict.get('class_name', '') == 'TrainedUncertaintyAwareClassifierNN':
            from ..models.pytorch import TrainedUncertaintyAwareClassifierNN
            model = TrainedUncertaintyAwareRegressorNN.from_config(config_dict)
            model.load_state_dict(state_dict)
            model.eval()
    else:
        print(f'Specified path, {model_path}, is not a PyTorch custom model file! Aborting!')
    return model


def save_model(model, model_path):
    model_save_dict = {
        'config_dict': model.get_config(),
        'state_dict': model.state_dict()
    }
    if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'state_dict'):
        model_save_dict['optim_dict'] = model.optimizer.state_dict()
    torch.save(model_save_dict, model_path)


