import psutil
from pathlib import Path
import numpy as np
import torch

from .helpers import numpy_default_dtype

torch.set_default_dtype(torch.float64 if numpy_default_dtype == np.float64 else torch.float32)
default_dtype = torch.get_default_dtype()
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_fuzz_factor(dtype):
    if dtype == torch.float16:
        return np.finfo(np.float16).eps
    elif dtype == torch.float32:
        return np.finfo(np.float32).eps
    elif dtype == torch.float64:
        return np.finfo(np.float64).eps
    else:
        return 0.0


def get_device_info(device_type=default_device):
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() and device_type == 'cuda' else psutil.cpu_count(logical=False)
    device_name = str(torch.device(device_type))
    return device_name, n_devices


def set_device_parallelism(intraop, interop=None):
    if isinstance(intraop, int):
        if not isinstance(interop, int):
            interop = intraop
        torch.set_num_threads(intraop)
        torch.set_num_interop_threads(interop)


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


def create_noise_contrastive_prior_loss_function(n_outputs, nll_weights, epi_weights, alea_weights, distance_loss, device=default_device, verbosity=0):
    if n_outputs > 1:
        from ..models.noise_contrastive_pytorch import MultiOutputNoiseContrastivePriorLoss
        return MultiOutputNoiseContrastivePriorLoss(n_outputs, nll_weights, epi_weights, alea_weights, distance_loss, reduction='sum', device=device)
    elif n_outputs == 1:
        from ..models.noise_contrastive_pytorch import NoiseContrastivePriorLoss
        return NoiseContrastivePriorLoss(nll_weights, epi_weights, alea_weights, distance_loss, reduction='sum', device=device)
    else:
        raise ValueError('Number of outputs to loss function generator must be an integer greater than zero.')


def create_evidential_loss_function(n_outputs, nll_weights, evi_weights, device=default_device, verbosity=0):
    if n_outputs > 1:
        from ..models.evidential_pytorch import MultiOutputEvidentialLoss
        return MultiOutputEvidentialLoss(n_outputs, nll_weights, evi_weights, reduction='sum', device=device)
    elif n_outputs == 1:
        from ..models.evidential_pytorch import EvidentialLoss
        return EvidentialLoss(nll_weights, evi_weights, reduction='sum', device=device)
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
    device=default_device,
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
        name=name,
        device=device
    )
    return model


def create_regressor_loss_function(n_outputs, style='ncp', device=default_device, verbosity=0, **kwargs):
    if style == 'ncp':
        return create_noise_contrastive_prior_loss_function(n_outputs, device=device, verbosity=verbosity, **kwargs)
    elif style == 'evidential':
        return create_evidential_loss_function(n_outputs, device=device, verbosity=verbosity, **kwargs)
    else:
        raise KeyError('Invalid loss function style passed to loss function generator.')


def wrap_regressor_model(model, scaler_in, scaler_out, device=default_device):
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
        name=f'wrapped_{model.name}',
        device=device
    )
    return wrapper


def create_classifier_model():
    return None


def create_classifier_loss_function():
    return None


def wrap_classifier_model(model, scaler_in, names_out, device=default_device):
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
        name=f'wrapped_{model.name}',
        device=device
    )
    return wrapper


def load_model(model_path, device=default_device):
    model = None
    mpath = Path(model_path)
    if mpath.is_file():
        model_save_dict = torch.load(mpath)
        config_dict = model_save_dict.get('config_dict', None)
        state_dict = model_save_dict.get('state_dict', None)
        if config_dict.get('class_name', 'TrainedUncertaintyAwareRegressorNN') == 'TrainedUncertaintyAwareRegressorNN':
            from ..models.pytorch import TrainedUncertaintyAwareRegressorNN
            model = TrainedUncertaintyAwareRegressorNN.from_config(config_dict)
            model.load_state_dict(state_dict)
            model = model.to(torch.device(device), default_dtype)
            model.eval()
        elif config_dict.get('class_name', '') == 'TrainedUncertaintyAwareClassifierNN':
            from ..models.pytorch import TrainedUncertaintyAwareClassifierNN
            model = TrainedUncertaintyAwareRegressorNN.from_config(config_dict)
            model.load_state_dict(state_dict)
            model = model.to(torch.device(device), default_dtype)
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


