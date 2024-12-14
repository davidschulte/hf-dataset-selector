from transformers import PretrainedConfig
from typing import Optional


class ESMConfig(PretrainedConfig):

    def __init__(
            self,
            base_model_name: Optional[str] = None,
            task_id: Optional[str] = None,
            task_subset: Optional[str] = None,
            text_column: Optional[str] = None,
            label_column: Optional[str] = None,
            task_split: Optional[str] = None,
            num_examples: Optional[int] = None,
            seed: Optional[int] = None,
            language: Optional[str] = None,
            esm_architecture: Optional[str] = None,
            lm_num_epochs: Optional[int] = None,
            lm_batch_size: Optional[int] = None,
            lm_learning_rate: Optional[float] = None,
            lm_weight_decay: Optional[float] = None,
            lm_optimizer: Optional[str] = None,
            esm_num_epochs: Optional[int] = None,
            esm_batch_size: Optional[int] = None,
            esm_learning_rate: Optional[float] = None,
            esm_weight_decay: Optional[float] = None,
            esm_optimizer: Optional[str] = None,
            developers: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.task_id = task_id
        self.task_subset = task_subset
        self.text_column = text_column
        self.label_column= label_column
        self.task_split = task_split
        self.num_examples = num_examples
        self.seed = seed
        self.language = language
        self.esm_architecture = esm_architecture
        self.lm_num_epochs = lm_num_epochs
        self.lm_batch_size = lm_batch_size
        self.lm_learning_rate = lm_learning_rate
        self.lm_weight_decay = lm_weight_decay
        self.lm_optimizer = lm_optimizer
        self.esm_num_epochs = esm_num_epochs
        self.esm_batch_size = esm_batch_size
        self.esm_learning_rate = esm_learning_rate
        self.esm_weight_decay = esm_weight_decay
        self.esm_optimizer = esm_optimizer
        self.developers = developers

        # assert self.is_valid

    @property
    def is_valid(self):
        return isinstance(self.base_model_name, str) and self.base_model_name and \
            isinstance(self.task_id, str) and self.task_id
