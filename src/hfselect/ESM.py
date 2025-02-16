import torch
from safetensors.torch import load_file, save_file
import torch.nn as nn
from typing import Dict, Optional, Union
from huggingface_hub import PyTorchModelHubMixin, create_repo, ModelCard, ModelCardData
import os
import warnings
from hfselect import logger
from .ESMConfig import ESMConfig, InvalidESMConfigError


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

        self.config = config or ESMConfig()
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.sequential = nn.Sequential(nn.Linear(768, 768))

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

        self.convert_legacy_to_new()
        if config is None:
            config = self.config

        if not config.is_valid:
            raise InvalidESMConfigError()

        self.push_to_hub(repo_id=repo_id)
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
            **config.to_dict()
        )
        card.push_to_hub(repo_id)

    @classmethod
    def from_disk(
            cls,
            filepath: str,
            device_name: str = "cpu",
    ) -> "ESM":
        if filepath.endswith(".pt"):
            device = torch.device(device_name)
            state_dict = torch.load(filepath, map_location=device)
        elif filepath.endswith(".safetensors"):
            state_dict = load_file(filepath)
        else:
            logger.warning(f"Unknown file extension in model filepath: .{filepath.split('.')[-1]}")
            state_dict = load_file(filepath)
        esm = ESM()
        esm.load_state_dict(state_dict)

        esm.convert_legacy_to_new()

        return esm

    def to_disk(self, filepath: str) -> None:
        self.convert_legacy_to_new()
        save_file(self.state_dict(), filepath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise ESMNotInitializedError()

        return self.model(x)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"ESM - Task ID: {self.config.get('task_id', 'N/A')} - Subset: {self.config.get('task_subset', 'N/A')}"

    def convert_legacy_to_new(self) -> None:
        # Convert legacy ESMs
        if hasattr(self, "sequential"):
            if self.model is None:
                self.model = self.sequential
                if (
                        isinstance(self.model, nn.Sequential)
                        and isinstance(self.model[0], nn.Linear)
                        and len(self.model) == 1
                ):
                    self.model = self.model[0]

            if isinstance(self.model, nn.Linear):
                self.architecture = "linear"
                self.embedding_dim = self.model.in_features

            else:
                warnings.warn("Could not determine ESM architecture while loading.")

            del self.sequential

    @property
    def is_initialized(self) -> bool:
        return self.model is not None
