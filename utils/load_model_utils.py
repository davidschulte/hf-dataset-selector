import torch
from esm import ESM
from config import GPU_DEVICE


def load_model(state_dict_path, model_type, device=None):
    if device is None:
        device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    state_dict = torch.load(state_dict_path, map_location=device)

    num_layers = int(len(state_dict) / 2)

    input_dim = state_dict['sequential.0.weight'].shape[1]
    # output_dim = state_dict[f'sequential.{num_layers - 1}.weight'].shape[0]
    output_dim = list(state_dict.items())[-1][1].shape[0]

    # optional_layer_dims = [state_dict[f'sequential.{i}.weight'].shape[0] for i in range(0, num_layers - 2, 2)]
    optional_layer_dims = [state_dict[key].shape[0] for key in state_dict.keys() if 'bias' in key][:-1]

    if model_type == 'transformation_network':
        model = ESM(optional_layer_dims=optional_layer_dims)
    else:
        print(f'Could not create model with type {model_type}.')
        return

    model.load_state_dict(state_dict)

    if device is not None:
        model.to(device)

    return model
