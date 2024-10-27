from abc import abstractmethod
from ESM import ESM
from transformers import get_linear_schedule_with_warmup, PreTrainedModel
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import AdamW
from src.hfselect.model_utils import create_sequence_classification_model, get_base_model
import os
from typing import Optional
import time
import json
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import SequentialSampler, DataLoader
from .embedding_dataset import EmbeddingDataset, create_embedding_dataset
from .dataset import Dataset


class Trainer:
    def __init__(self,
                 model: Optional[nn.Module] = None,
                 optimizer=None,
                 learning_rate=0.001,
                 weight_decay=0.01,
                 scheduler=None,
                 device: str = "cpu"):
        self.output_dim = output_dim
        self.model = self._create_model() if model is None else model
        self.optimizer = self._create_optimizer(self.model, weight_decay, learning_rate=learning_rate)\
            if optimizer is None else optimizer
        # self.scheduler = self._create_scheduler(self.optimizer, num_train_steps=num_train_steps) if scheduler is None \
        #     else scheduler
        self.scheduler = scheduler

        if device != "cpu" and torch.cuda.is_available():
            self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = "cpu"
        # print(self.device)
        self.model.to(self.device)

        self.total_loss = 0
        self.num_train_examples = 0

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    @staticmethod
    def _create_optimizer(model, weight_decay, learning_rate=0.001):
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    @staticmethod
    def _create_scheduler(optimizer, num_train_steps):
        return get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=num_train_steps)

    def reset_loss(self):
        # print(f'Loss: {self.total_loss}')
        self.total_loss = 0
        self.num_train_examples = 0

    @property
    def avg_loss(self):
        return self.total_loss / self.num_train_examples


class TransformerTrainer(Trainer):

    def __init__(self, output_dim, model=None, optimizer=None, weight_decay=0.01, scheduler=None,
                 freeze_base_model=False, device="cpu"):
        super(TransformerTrainer, self).__init__(model=model,
                                                 optimizer=optimizer,
                                                 weight_decay=weight_decay,
                                                 scheduler=scheduler,
                                                 learning_rate=2e-5,
                                                 device=device)

        self.output_dim = output_dim

        self.freeze_base_model = freeze_base_model

        if self.freeze_base_model:
            base_model = get_base_model(self.model)
            for param in base_model.parameters():
                param.requires_grad = False

    def train_step(self, batch):
        batch = tuple(b.to(self.device) for b in batch)
        b_input_ids, b_input_mask, b_labels = batch

        self.model.train()
        self.model.zero_grad()

        outputs = self.model(b_input_ids,
                             # token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = outputs[0]
        self.total_loss += loss.item()
        self.num_train_examples += len(b_input_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _create_model(self):
        return create_sequence_classification_model(num_labels=self.output_dim)


class ESMTrainer(Trainer):
    def __init__(self, model=None, model_optional_layer_dims=None,
                 optimizer=None, weight_decay=0.01, learning_rate=0.001, scheduler=None, device="cpu"):

        self.model_optional_layer_dims = model_optional_layer_dims

        super(ESMTrainer, self).__init__(model=model,
                                         optimizer=optimizer,
                                         weight_decay=weight_decay,
                                         scheduler=scheduler,
                                         learning_rate=learning_rate,
                                         device=device)

        self.loss_fct = MSELoss()

    def _create_model(self):
        return ESM(optional_layer_dims=self.model_optional_layer_dims)

    def train_step(self, embeddings_batch):
        self.model.train()

        embeddings_batch = tuple(b.to(self.device) for b in embeddings_batch)
        b_standard_embeddings, b_transferred_embeddings = embeddings_batch

        self.model.zero_grad()
        outputs = self.model(b_standard_embeddings.float())
        # loss = self.loss_fct(outputs, b_labels.float())
        loss = self.loss_fct(outputs, b_transferred_embeddings.float())
        self.total_loss += loss.item()
        self.num_train_examples += len(b_standard_embeddings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_with_embeddings(self,
                              embeddings_dataset: EmbeddingDataset,
                              output_filepath: str = None,
                              num_epochs: int = 10,
                              batch_size: int = 32,
                              overwrite:bool = False):

        if output_filepath:
            if os.path.isfile(output_filepath) and not overwrite:
                print('Found transformation network on disk.')
                return ESM.from_pretrained(output_filepath)

        sampler = SequentialSampler(embeddings_dataset)
        dataloader = DataLoader(embeddings_dataset, sampler=sampler, batch_size=batch_size)

        num_train_steps = len(dataloader) * num_epochs

        # TODO: Check scheduler if scheduler should always be overwritten
        if self.scheduler is None:
            self.scheduler = self._create_scheduler(optimizer=self.optimizer, num_train_steps=num_train_steps)

        epoch_train_durations = []
        epoch_avg_losses = []
        start_time = time.time()
        for epoch_i in range(num_epochs):

            self.reset_loss()
            with tqdm(dataloader, desc=f'Training: Epoch {epoch_i} / {num_epochs}', unit='batch') as pbar:

                for step, batch in enumerate(pbar):
                    loss = self.train_step(batch)

                    avg_train_loss = loss / batch_size

                    pbar.set_postfix(avg_train_loss=avg_train_loss)

            end_time = time.time()
            epoch_train_durations.append(end_time - start_time)
            start_time = end_time
            epoch_avg_losses.append(self.avg_loss)

        if output_filepath:
            output_dir = os.path.dirname(output_filepath)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), output_filepath)
            train_info_dict = {
                'training_completed_timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                'num_epochs': num_epochs,
                'num_train_examples': len(embeddings_dataset),
                'epoch_train_durations': epoch_train_durations,
                'epoch_avg_losses': epoch_avg_losses
            }

            with open(os.path.join(output_dir, 'train_info.json'), 'w') as f:
                json.dump(train_info_dict, f)

            print('Saved model.')

        return self.model

    def train_with_models(self,
                          dataset: Dataset,
                          base_model: PreTrainedModel,
                          tuned_model: PreTrainedModel,
                          model_output_filepath: Optional[str] = None,
                          embeddings_output_filepath: Optional[str] = None,
                          num_epochs: int = 10,
                          train_batch_size: int = 32,
                          embeddings_batch_size: int = 128,
                          device_name: str = "cpu",
                          overwrite_model: bool = False,
                          overwrite_embeddings: bool = False):
        embeddings_dataset = create_embedding_dataset(dataset=dataset,
                                                      base_model=base_model,
                                                      tuned_model=tuned_model,
                                                      batch_size=embeddings_batch_size,
                                                      output_path=embeddings_output_filepath,
                                                      device_name=device_name,
                                                      overwrite=overwrite_embeddings)

        return self.train_with_embeddings(embeddings_dataset=embeddings_dataset,
                                          output_filepath=model_output_filepath,
                                          num_epochs=num_epochs,
                                          batch_size=train_batch_size,
                                          overwrite=overwrite_model)
