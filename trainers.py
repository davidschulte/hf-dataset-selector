from abc import abstractmethod
from esm import ESM
from transformers import get_linear_schedule_with_warmup, logging, BertForSequenceClassification
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from utils.model_utils import create_sequence_classification_model, get_base_model
from config import GPU_DEVICE


class Trainer:
    def __init__(self, output_dim, num_train_steps, model=None,
                 optimizer=None, learning_rate=0.001, weight_decay=0.01, scheduler=None):
        self.output_dim = output_dim
        self.model = self._create_model() if model is None else model
        self.optimizer = self._create_optimizer(self.model, weight_decay, learning_rate=learning_rate)\
            if optimizer is None else optimizer
        self.scheduler = self._create_scheduler(self.optimizer, num_train_steps=num_train_steps) if scheduler is None \
            else scheduler

        self.device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
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

    def __init__(self, output_dim, num_train_steps, model=None, optimizer=None, weight_decay=0.01, scheduler=None,
                 freeze_base_model=False):
        super(TransformerTrainer, self).__init__(output_dim=output_dim,
                                                 num_train_steps=num_train_steps,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 weight_decay=weight_decay,
                                                 scheduler=scheduler,
                                                 learning_rate=2e-5)

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
    def __init__(self, num_train_steps, model=None, model_optional_layer_dims=None,
                 optimizer=None, weight_decay=0.01, learning_rate=0.001, scheduler=None):

        self.model_optional_layer_dims = model_optional_layer_dims

        super(ESMTrainer, self).__init__(output_dim=None,
                                         num_train_steps=num_train_steps,
                                         model=model,
                                         optimizer=optimizer,
                                         weight_decay=weight_decay,
                                         scheduler=scheduler,
                                         learning_rate=learning_rate)

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
