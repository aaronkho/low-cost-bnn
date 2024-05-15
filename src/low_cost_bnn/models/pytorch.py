import torch
from torch.nn import Module, ModuleDict, Linear, LeakyReLU, SoftPlus
import torch.distributions as tnd


class EpistemicLayer(Module):

    def __init__(self):
        super(EpistemicLayer, self).__init__()


class LowCostBNN(Module):

    def __init__(self, n_inputs, n_outputs, n_hidden, n_specialized=None):
        super(LowCostBNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        activation = LeakyReLU(negative_slope=0.2)

        n_special = [n_hidden] * self.n_outputs
        if isinstance(n_specialized, (list, tuple)):
            for ii in range(n_outputs):
                n_special[ii] = n_specialized[ii] if ii < len(n_specialized) else n_specialized[-1]

        self.commons = Linear(n_hidden)
        self.commons_act = activation
        self.specials = ModuleDict()
        for ii in range(n_outputs):
            self.specials.update({f'specialized{ii}': Linear(

    def forward(self, inputs):
        commons = self.commons(inputs)
        commons = self.commons_act(commons)
