from abc import abstractmethod
from .ESM import ESM
from .ESMConfig import ESMConfig
from transformers import get_linear_schedule_with_warmup, PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn
from torch.optim import AdamW
import os
from typing import Optional, List, Tuple, Union
import time
import json
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import RandomSampler, DataLoader
from .embedding_dataset import EmbeddingDataset, create_embedding_dataset
from .dataset import Dataset
import warnings


class Trainer:
    def __init__(
            self,
            model: Optional[nn.Module] = None,
            optimizer: Optional["torch.optim.Optimizer"] = None,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            device_name: str = "cpu"
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = None

        if device_name != "cpu" and torch.cuda.is_available():
            self.device = torch.device(device_name) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = "cpu"

        # self.model.to(self.device)

        self.total_loss = 0
        self.num_train_examples = 0

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def _create_optimizer(self, model: nn.Module) -> AdamW:
        return AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def _create_scheduler(
            optimizer: "torch.optim.Optimizer",
            num_train_steps: int
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        return get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=num_train_steps)

    def reset_loss(self):
        self.total_loss = 0
        self.num_train_examples = 0

    @property
    def avg_loss(self):
        return self.total_loss / self.num_train_examples


class ESMTrainer(Trainer):
    def __init__(
            self,
            model: Optional[nn.Module] = None,
            model_optional_layer_dims: Optional[List[int]] = None,
            optimizer: Optional["torch.optim.Optimizer"] = None,
            weight_decay: float = 0.01,
            learning_rate: float = 0.001,
            device_name: str = "cpu"
    ):

        self.model_optional_layer_dims = model_optional_layer_dims

        super(ESMTrainer, self).__init__(
            model=model,
            optimizer=optimizer,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            device_name=device_name
        )

        self.loss_fct = nn.MSELoss()

    def _create_model(
            self,
            architecture: Optional[Union[str, dict[str, Union[str, tuple[str]]]]] = None,
            embedding_dim: Optional[int] = None,
    ) -> ESM:
        return ESM(architecture=architecture, embedding_dim=embedding_dim)

    def train_step(self, embeddings_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
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
            architecture: Optional[Union[str, dict[str, Union[str, tuple[str]]]]] = 'linear',
            output_filepath: str = None,
            num_epochs: int = 10,
            batch_size: int = 32,
            verbose: int = 1
    ) -> ESM:

        if self.model is None:
            self.model = self._create_model(architecture=architecture, embedding_dim=embedding_dataset.embedding_dim)

        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer = self._create_optimizer(model=self.model)

        sampler = RandomSampler(embedding_dataset)
        dataloader = DataLoader(embedding_dataset, sampler=sampler, batch_size=batch_size)

        num_train_steps = len(dataloader) * num_epochs

        self.scheduler = self._create_scheduler(optimizer=self.optimizer, num_train_steps=num_train_steps)

        epoch_train_durations = []
        epoch_avg_losses = []
        start_time = time.time()
        with tqdm(
                range(num_epochs),
                desc="Training ESM",
                unit="epoch",
                disable=verbose < 1
        ) as epoch_pbar:

            for epoch_i in epoch_pbar:

                self.reset_loss()

                with tqdm(
                        dataloader,
                        desc=f'Training: Epoch {epoch_i} / {num_epochs}',
                        unit='batch',
                        disable=verbose < 2
                ) as batch_pbar:

                    for batch in batch_pbar:
                        loss = self.train_step(batch)

                        avg_train_loss = loss / batch_size

                        epoch_pbar.set_postfix(avg_train_loss=avg_train_loss)
                        batch_pbar.set_postfix(avg_train_loss=avg_train_loss)

                end_time = time.time()
                epoch_train_durations.append(end_time - start_time)
                start_time = end_time
                epoch_avg_losses.append(self.avg_loss)

        self.model.config = ESMConfig(
            esm_num_epochs=num_epochs,
            esm_learning_rate=self.learning_rate,
            esm_weight_decay=self.weight_decay,
            esm_batch_size=batch_size
        )

        if output_filepath:
            output_dir = os.path.dirname(output_filepath)
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(output_filepath):
                warnings.warn(f"Overwriting ESM at path: {output_filepath}")

            torch.save(self.model.state_dict(), output_filepath)
            train_info_dict = {
                'training_completed_timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                'num_epochs': num_epochs,
                'num_train_examples': len(embedding_dataset),
                'epoch_train_durations': epoch_train_durations,
                'epoch_avg_losses': epoch_avg_losses
            }

            with open(os.path.join(output_dir, 'train_info.json'), 'w') as f:
                json.dump(train_info_dict, f)

            print('Saved model.')

        return self.model

    def train_with_models(
            self,
            dataset: Dataset,
            base_model: PreTrainedModel,
            tuned_model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            architecture: Optional[Union[str, dict[str, Union[str, tuple[str]]]]] = 'linear',
            model_output_filepath: Optional[str] = None,
            embeddings_output_filepath: Optional[str] = None,
            num_epochs: int = 10,
            train_batch_size: int = 32,
            embeddings_batch_size: int = 128,
            device_name: str = "cpu",
    ) -> ESM:
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
            output_filepath=model_output_filepath,
            num_epochs=num_epochs,
            batch_size=train_batch_size,
        )

        esm.config.update({**{"base_model_name": base_model.config.name_or_path}, **dataset.metadata})

        return esm
