from typing import Optional, Any, Union
from transformers import PretrainedConfig
from .version import __version__


def _format_text_column_names(text_column: Union[str, tuple]):
    # If multiple text columns are used as input, they have to be formatted to be displayed correctly.

    if isinstance(text_column, str):
        return text_column
    elif isinstance(text_column, tuple):
        return ",".join(text_column)
    else:
        return NotImplementedError(
            f"Can not format text column(s) of type {type(text_column)}."
        )


class InvalidESMConfigError(Exception):
    """
    Raised when the ESMConfig is invalid
    """

    default_message = "The Config is not a valid ESM Config. Task ID and base model name need to be specified."

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.default_message)


class ESMConfig(PretrainedConfig):
    """
    ESMConfig is a config for an ESM. It contains metadata that is parsed to the model card when uploaded to HF.
    """

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        task_id: Optional[str] = None,
        task_subset: Optional[str] = None,
        text_column: Optional[Union[str, tuple[str]]] = None,
        label_column: Optional[str] = None,
        task_split: Optional[str] = None,
        num_examples: Optional[int] = None,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        esm_architecture: Optional[str] = None,
        esm_embedding_dim: Optional[int] = None,
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
        version: Optional[str] = __version__,
        **kwargs,
    ):
        """
        Creates a new ESMConfig

        Args:
            base_model_name: The name of the base model. Use its repo-id, e.g. google-bert/bert-base-uncased.
            task_id: The dataset ID of the task used for fine-tuning the base model
            task_subset: The HF subset of the dataset
            text_column: The name of the text column(s) used in the task
            label_column: The name of the label column
            task_split: The split used for training the ESM
            num_examples: The number of tutorials samples from the dataset to train the ESM
            seed: The random state used for sampling tutorials from the dataset to train the ESM
            language: The langauge of the dataset used to train the ESM
            esm_architecture: The architecture of the ESM
            esm_embedding_dim: The embedding dimension of the language model
            lm_num_epochs: The number of epochs used for fine-tuning the language model
            lm_batch_size: The batch size used for fine-tuning the language model
            lm_learning_rate: The learning rate used for fine-tuning the language model
            lm_weight_decay: The weight decay used for fine-tuning the language model
            lm_optimizer: The optimizer used for fine-tuning the language model
            esm_num_epochs: The number of epochs used for training the ESM
            esm_batch_size: The batch size used for training the ESM
            esm_learning_rate: The learning rate used for training the ESM
            esm_weight_decay: The weight decay used for training the ESM
            esm_optimizer: The optimizer used for training the ESM
            developers: The name of the developer(s) that trained the ESM
            **kwargs: Optional additional metadata
        """
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.task_id = task_id
        self.task_subset = task_subset
        self.text_column = text_column
        self.label_column = label_column
        self.task_split = task_split
        self.num_examples = num_examples
        self.seed = seed
        self.language = language
        self.esm_architecture = esm_architecture
        self.esm_embedding_dim = esm_embedding_dim
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
        self.version = version

    @property
    def is_valid(self) -> bool:
        """
        Checks if the config is valid. Only ESMs with valid configs should be uploaded and used for task selection.
        An ESMConfig must contain the name of base langauge model and the dataset that was used to fine-tune it.

        Returns:
            The validity of the config
        """
        return (
            self.base_model_name
            and isinstance(self.base_model_name, str)
            and self.task_id
            and isinstance(self.task_id, str)
        )

    def __str__(self) -> str:
        return (
            f"ESMConfig Task ID: {self.task_id:<50} Task Subset: {self.task_id:<50} Task Split: {self.task_split:<10}"
            f"Text Column: {_format_text_column_names(self.text_column)}"
            f"Label Column: {self.label_column:<10} Num Examples: {self.num_examples}"
        )

    def get(self, attr_name: str, default_return_val: Any = None) -> Any:
        """
        A get function to make the class behave like a dictionary

        Args:
            attr_name: The name of the attribute to access
            default_return_val: A default value that gets returned when the attribute does not exist

        Returns:
            The value of the attribute if it exists, and otherwise the default return value
        """
        return self.__dict__.get(attr_name, default_return_val)
