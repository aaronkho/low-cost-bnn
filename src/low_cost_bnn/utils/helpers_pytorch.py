import numpy as np
import torch

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

