from typing import Optional, Union
from abc import ABC, abstractmethod
import os
from datetime import datetime
import time
import json

from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .esm import ESM
from .esmconfig import ESMConfig
from .embedding_dataset import EmbeddingDataset, create_embedding_dataset
from .dataset import Dataset
from hfselect import logger


class Trainer(ABC):
    """
    A abstract trainer class
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device_name: str = "cpu",
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = None

        if device_name != "cpu" and torch.cuda.is_available():
            self.device = (
                torch.device(device_name)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = "cpu"

        # self.model.to(self.device)

        self.total_loss = 0
        self.num_train_examples = 0

    @abstractmethod
    def _train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def _create_optimizer(self, model: nn.Module) -> AdamW:
        return AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    @staticmethod
    def _create_scheduler(
        optimizer: "torch.optim.Optimizer", num_train_steps: int
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

    def reset_loss(self):
        """
        Resets the loss for optimization.

        Returns:

        """
        self.total_loss = 0
        self.num_train_examples = 0

    @property
    def avg_loss(self):
        """
        The average loss per training example

        Returns:
            The average loss per training example
        """
        return self.total_loss / self.num_train_examples


class ESMTrainer(Trainer):
    """
    A trainer class that fabricates ESMs
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        weight_decay: float = 0.01,
        learning_rate: float = 0.01,
        device_name: str = "cpu",
    ):
        """
        Creates an ESMTrainer

        Args:
            model: The underlying model to be used in the ESM
            optimizer: The optimizer for training the ESM
            weight_decay: The weight decay for training the ESM
            learning_rate: The learning rate for training the ESM
            device_name: The device name of the device for computation (e.g. "cpu", "cuda")
        """

        super(ESMTrainer, self).__init__(
            model=model,
            optimizer=optimizer,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            device_name=device_name,
        )

        self.loss_fct = nn.MSELoss()

    def _create_model(
        self,
        architecture: Optional[Union[str, dict[str, Union[str, tuple[str]]]]] = None,
        embedding_dim: Optional[int] = None,
    ) -> ESM:
        # Creates a new ESM
        return ESM(architecture=architecture, embedding_dim=embedding_dim)

    def _train_step(self, embeddings_batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        # One train step for one batch
        self.model.train()

        embeddings_batch = tuple(b.to(self.device) for b in embeddings_batch)
        b_standard_embeddings, b_transferred_embeddings = embeddings_batch

        self.model.zero_grad()
        outputs = self.model(b_standard_embeddings.float())
        loss = self.loss_fct(outputs, b_transferred_embeddings.float())
        self.total_loss += loss.item()
        self.num_train_examples += len(b_standard_embeddings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_with_embeddings(
        self,
        embedding_dataset: EmbeddingDataset,
        architecture: Optional[
            Union[str, dict[str, Union[str, tuple[str]]]]
        ] = "linear",
        output_dir: Optional[str] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        reset_model: bool = True,
        verbose: int = 1,
    ) -> ESM:
        """
        Trains an ESM using an EmbeddingDataset dataset. The ESM is fitted to the embedding pairs in the dataset.

        Args:
            embedding_dataset: The embeddings of the same dataset embedded by a base model and a fine-tuned model
            architecture: The desired architecture of the ESM
            output_dir: If a directory is specified, the ESM will be saved locally after training
            num_epochs: The number of epochs for training the ESM
            batch_size: The batch size for training the ESM
            reset_model: If set to False, the same model with be trained further with multiple calls of the function.
            verbose: 0 hides everything, 1 shows the complete training of the ESM, and 2 shows the ESM training epochs.

        Returns:
            The resulting ESM
        """
        if self.model is None or reset_model:
            self.model = self._create_model(
                architecture=architecture, embedding_dim=embedding_dataset.embedding_dim
            )

        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer = self._create_optimizer(model=self.model)

        sampler = RandomSampler(embedding_dataset)
        dataloader = DataLoader(
            embedding_dataset, sampler=sampler, batch_size=batch_size
        )

        num_train_steps = len(dataloader) * num_epochs

        self.scheduler = self._create_scheduler(
            optimizer=self.optimizer, num_train_steps=num_train_steps
        )

        epoch_train_durations = []
        epoch_avg_losses = []
        with tqdm(
            range(num_epochs), desc="Training ESM", unit="epoch", disable=verbose < 1
        ) as epoch_pbar:
            for epoch_i in epoch_pbar:
                self.reset_loss()

                start_time = time.perf_counter()
                with tqdm(
                    dataloader,
                    desc=f"Training: Epoch {epoch_i} / {num_epochs}",
                    unit="batch",
                    disable=verbose < 2,
                ) as batch_pbar:
                    for batch in batch_pbar:
                        loss = self._train_step(batch)

                        avg_train_loss = loss / batch_size

                        epoch_pbar.set_postfix(avg_train_loss=avg_train_loss)
                        batch_pbar.set_postfix(avg_train_loss=avg_train_loss)

                end_time = time.perf_counter()
                epoch_train_durations.append(end_time - start_time)
                epoch_avg_losses.append(self.avg_loss)

        self.model.config = ESMConfig(
            esm_num_epochs=num_epochs,
            esm_learning_rate=self.learning_rate,
            esm_weight_decay=self.weight_decay,
            esm_batch_size=batch_size,
            esm_architecture=architecture,
            esm_embedding_dim=embedding_dataset.embedding_dim,
        )
        self.model.config.update(embedding_dataset.metadata)

        if output_dir:
            if os.path.isdir(output_dir):
                logger.warning(f"Overwriting ESM at path: {output_dir}")

            self.model.save_pretrained(output_dir)

            train_info_dict = {
                "training_completed_timestamp": datetime.now().strftime(
                    "%m/%d/%Y, %H:%M:%S"
                ),
                "num_epochs": num_epochs,
                "num_train_examples": len(embedding_dataset),
                "epoch_train_durations": epoch_train_durations,
                "epoch_avg_losses": epoch_avg_losses,
            }

            with open(os.path.join(output_dir, "train_info.json"), "w") as f:
                json.dump(train_info_dict, f)

        return self.model

    def train_with_models(
        self,
        dataset: Dataset,
        base_model: PreTrainedModel,
        tuned_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        architecture: Optional[
            Union[str, dict[str, Union[str, tuple[str]]]]
        ] = "linear",
        model_output_dir: Optional[str] = None,
        embeddings_output_filepath: Optional[str] = None,
        num_epochs: int = 10,
        train_batch_size: int = 32,
        embeddings_batch_size: int = 128,
        device_name: str = "cpu",
    ) -> ESM:
        """
        Trains an ESM using a dataset, a base language model and a fine-tuned language model.
        Internally, an EmbeddingDataset is created. Following this, the train_with_embeddings is called
        and the ESM is fitted to the embedding pairs in the dataset.

        Args:
            dataset: The dataset used for fine-tuning the language model
            base_model: The base language model
            tuned_model: The fine-tuned language model
            tokenizer: The tokenizer for processing input texts
            architecture: The desired architecture of the ESM
            model_output_dir: If a directory is specified, the ESM will be saved locally after training
            embeddings_output_filepath: If a filepath is specified, the EmbeddingDataset will be saved locally
            num_epochs: The number of epochs for training the ESM
            train_batch_size: The batch size for training the ESM
            embeddings_batch_size: The batch size for creating the EmbeddingDataset
            device_name: The device name of the device for computation (e.g. "cpu", "cuda")

        Returns:
            The resulting ESM
        """
        embedding_dataset = create_embedding_dataset(
            dataset=dataset,
            base_model=base_model,
            tuned_model=tuned_model,
            tokenizer=tokenizer,
            batch_size=embeddings_batch_size,
            device_name=device_name,
        )

        if embeddings_output_filepath:
            embedding_dataset.save(embeddings_output_filepath)

        esm = self.train_with_embeddings(
            embedding_dataset=embedding_dataset,
            architecture=architecture,
            output_dir=model_output_dir,
            num_epochs=num_epochs,
            batch_size=train_batch_size,
        )

        return esm
