import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, HfApi, create_repo, ModelCard, ModelCardData
from tqdm import tqdm
import os
from .ESMConfig import ESMConfig
# from . import hf_api


class ESM(nn.Module, PyTorchModelHubMixin):

    def __init__(
            self,
            embedding_dim: int = 768,
            optional_layer_dims: Optional[List[int]] = None,
            config: Union[ESMConfig, Dict[str, Union[float, int, str]]] = None
    ):
        super(ESM, self).__init__()
        if isinstance(optional_layer_dims, int):
            optional_layer_dims = [optional_layer_dims]
        if optional_layer_dims is None:
            self.layer_dims = [embedding_dim, embedding_dim]
            # layers = [nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)]
        else:
            self.layer_dims = [embedding_dim] + optional_layer_dims + [embedding_dim]

        self.embedding_dim = embedding_dim
        self.relu = nn.ReLU()

        # layers = [nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(dims, dims[1:])]
        linear_layers = [nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(self.layer_dims, self.layer_dims[1:])]

        layers = []
        for ll in linear_layers:
            layers += [ll, self.relu]
        layers = layers[:-1]

        # TODO: Fix bottleneck idx after relu insertion
        self.bottleneck_idx = np.argmin(self.layer_dims[1:]) + 1
        self.bottleneck_dim = np.min(self.layer_dims)
        self.sequential = nn.Sequential(*layers)

        self.config = config

    def publish(
            self,
            repo_id: str,
            config: Optional[Union[ESMConfig, Dict[str, Union[float, int, str]]]] = None
    ) -> None:
        create_repo(repo_id=repo_id, exist_ok=True)

        if config is None:
            config = self.config

        assert config.is_valid

        self.push_to_hub(repo_id=repo_id)#, config=config)
        config.push_to_hub(repo_id=repo_id)

        card_data = ModelCardData(license='apache-2.0',
                                  datasets=[config.task_id],
                                  base_model=config.base_model_name,
                                  tags=["embedding_space_map", f"BaseLM:{config.base_model_name}"])

        card = ModelCard.from_template(
            card_data,
            template_path=os.path.join(os.path.dirname(__file__), "modelcard_template.md"),
            model_id=config.task_id,
            model_description="ESM",
            # datasets=[self.task_id],
            **config.to_dict()
        )
        card.push_to_hub(repo_id)

    @classmethod
    def from_disk(
            cls,
            filepath: str,
            device_name: str = "cpu",
    ) -> "ESM":

        device = torch.device(device_name)
        state_dict = torch.load(filepath, map_location=device)
        embedding_dim = state_dict['sequential.0.weight'].shape[1]

        esm = ESM(embedding_dim=embedding_dim)
        esm.load_state_dict(state_dict)

        return esm

    @classmethod
    def from_hugging_face(cls, repo_id: str) -> "ESM":
        esm = ESM.from_disk(hf_hub_download(repo_id, filename="model.safetensors"))
        esm.repo_id = repo_id

        return esm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential[:self.bottleneck_idx](x)

    @property
    def bottleneck_model(self) -> nn.Sequential:
        return self.sequential[:self.bottleneck_idx]


def find_esm_repo_ids(model_name: str) -> List[str]:
    hf_api = HfApi()
    model_infos = hf_api.list_models(filter=["embedding_space_map", f"BaseLM:{model_name}"])

    return [model_info.id for model_info in model_infos]


def fetch_esms(model_name: str) -> dict[str, "ESM"]:
    repo_ids = find_esm_repo_ids(model_name=model_name)

    esms = {}
    for repo_id in tqdm(repo_ids):
        try:
            esms[repo_id] = ESM.from_pretrained(repo_id)
        except Exception as e:
            pass

    return esms
