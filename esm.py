import torch.nn as nn
import numpy as np
from utils.model_utils import EMBEDDING_SIZE

class ESM(nn.Module):

    def __init__(self, optional_layer_dims=None):
        super(ESM, self).__init__()
        if isinstance(optional_layer_dims, int):
            optional_layer_dims = [optional_layer_dims]
        if optional_layer_dims is None:
            dims = [EMBEDDING_SIZE, EMBEDDING_SIZE]
            # layers = [nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)]
        else:
            dims = [EMBEDDING_SIZE] + optional_layer_dims + [EMBEDDING_SIZE]

        self.relu = nn.ReLU()

        # layers = [nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dims, dims[1:])]
        linear_layers = [nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dims, dims[1:])]

        layers = []
        for ll in linear_layers:
            layers += [ll, self.relu]
        layers = layers[:-1]

        # TODO: Fix bottleneck idx after relu insertion
        self.bottleneck_idx = np.argmin(dims[1:]) + 1
        self.bottleneck_dim = np.min(dims)
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

    def forward_bottleneck(self, x):
        return self.sequential[:self.bottleneck_idx](x)

    @property
    def bottleneck_model(self):
        return self.sequential[:self.bottleneck_idx]
