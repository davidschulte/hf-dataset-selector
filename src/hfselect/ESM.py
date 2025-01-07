import torch
from safetensors.torch import load_file
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, create_repo, ModelCard, ModelCardData
import os
import warnings
from .ESMConfig import ESMConfig, InvalidESMConfigError
# from . import hf_api


class ESMNotInitializedError(Exception):

    custom_message = """
    ESM was not initialized correctly. Define the ESM architecture before using it for training or inference.
    """

    def __init__(self, details_message: Optional[str] = None):
        super().__init__(self.custom_message + details_message if details_message else self.custom_message)


class ESM(nn.Module, PyTorchModelHubMixin):

    def __init__(
            self,
            architecture: Optional[Union[str, dict[str, Union[str, tuple[str]]]]] = None,
            embedding_dim: Optional[int] = None,
            config: Optional[Union[ESMConfig, Dict[str, Union[float, int, str]]]] = None
    ):
        super(ESM, self).__init__()

        self.config = config
        self.architecture = architecture
        self.embedding_dim = embedding_dim

        if not self.architecture:
            self.model = None
        else:
            if self.architecture == "linear":
                if embedding_dim is None:
                    raise ESMNotInitializedError(details_message="Embedding dimension not provided.")

                self.model = nn.Linear(self.embedding_dim, self.embedding_dim)
            else:
                raise NotImplementedError(f"Could not create ESM with custom architecture: {self.architecture}")

    def publish(
            self,
            repo_id: str,
            config: Optional[Union[ESMConfig, Dict[str, Union[float, int, str]]]] = None
    ) -> None:
        create_repo(repo_id=repo_id, exist_ok=True)

        if config is None:
            config = self.config

        if not config.is_valid:
            raise InvalidESMConfigError()

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

        # device = torch.device(device_name)
        state_dict = load_file(filepath)
        # embedding_dim = state_dict['sequential.0.weight'].shape[1]

        esm = ESM().load_state_dict(state_dict)

        # Convert legacy ESMs
        if not hasattr(esm, "model") and hasattr(esm, "sequential"):
            esm.model = esm.sequential
            del esm.sequential
            if isinstance(esm.model, nn.Sequential) and isinstance(esm.model[0], nn.Linear) and len(esm.model) == 1:
                esm.model = esm.model[0]

        if isinstance(esm.model, nn.Linear):
            esm.architecture = "linear"
            esm.embedding_dim = esm.model.in_features

        else:
            warnings.warn("Could not determine ESM architecture while loading.")
        return esm

    @classmethod
    def from_hugging_face(cls, repo_id: str) -> "ESM":
        esm = ESM.from_disk(hf_hub_download(repo_id, filename="model.safetensors"))
        esm.repo_id = repo_id

        return esm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ESMNotInitializedError()

        return self.model(x)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"ESM - Task ID: {self.config.get('task_id', 'N/A')} - Subset: {self.config.get('task_subset', 'N/A')}"
