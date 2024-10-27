from torch.utils.data import Dataset as TorchDataset
import torch
from functools import partial
from datasets import load_dataset, ClassLabel, IterableDataset
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
import numpy as np
from typing import Optional, Union, List, Tuple

DATASET_STREAMING_BUFFER_SIZE = 100000
TEXT_SEPARATOR = ' [SEP] '


class EmptyDatasetError(Exception):
    pass


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def concat_columns(inputs, column_list: tuple):
    if isinstance(inputs, str):
        return TEXT_SEPARATOR.join([inputs[col] for col in column_list])

    return [TEXT_SEPARATOR.join([row[col] for col in column_list]) for row in inputs]


class Dataset(TorchDataset):

    def __init__(
            self,
            dataset: Union[HFDataset, IterableDataset],
            text_col: Union[str, Tuple[str]],
            label_col: str,
            is_regression: bool,
            metadata: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.text_col = text_col
        self.label_col = label_col
        self.is_regression = is_regression

        self.dataset_len = len(self.dataset)

        if self.dataset_len == 0:
            raise EmptyDatasetError(f'Dataset is empty.')

        label_features = self.dataset.features[label_col]

        self.has_string_labels = label_features.dtype == 'string'
        if self.has_string_labels:
            if isinstance(dataset, IterableDataset):
                label_list = sorted(list(set([example[label_col] for example in self.dataset])))
            else:
                label_list = sorted(list(set(self.dataset[label_col])))
            self.label_dim = len(label_list)
            self.class_label = ClassLabel(num_classes=self.label_dim, names=label_list)
        elif 'float' in label_features.dtype:
            self.label_dim = 1
            self.class_label = None
        else:
            try:
                self.label_dim = label_features.num_classes
            except:
                self.label_dim = np.max(self.dataset[label_col]) + 1
            self.class_label = None

        self.metadata = metadata

    @classmethod
    def from_hugging_face(
            cls,
            name: str,
            split: str,
            text_col: Union[str, List[str]],
            label_col: str,
            is_regression: bool,
            subset: Optional[str] = None,
            num_examples: Optional[int] = None,
            seed: Optional[int] = None,
            streaming: bool = False
            # load_locally: bool = True
    ) -> "Dataset":
        # if load_locally:
        #     dataset_path = get_dataset_filepath(dataset_name=dataset_name,
        #                                         split=split,
        #                                         max_num_examples=max_num_examples,
        #                                         seed=seed)
        #     if os.path.isdir(dataset_path):
        #         return load_from_disk(dataset_path)

        if subset is None:
            dataset = load_dataset(name, split=split, streaming=streaming)  # [12250:12750]
        else:
            dataset = load_dataset(name, subset, split=split, streaming=streaming)  # [12250:12750]

        cols_to_keep = text_col + [label_col] if isinstance(text_col, list) else [text_col, label_col]
        dataset = dataset.select_columns(cols_to_keep)

        task_type = "regression" if is_regression else "classification"
        if task_type == 'classification':
            dataset = dataset.filter(lambda example: example[label_col] != -1)

        if num_examples is not None:
            if streaming:
                dataset = dataset.shuffle(seed=seed, buffer_size=DATASET_STREAMING_BUFFER_SIZE).take(num_examples)
                dataset = HFDataset.from_generator(
                    partial(gen_from_iterable_dataset, dataset),
                    features=dataset.features
                )
            else:
                num_examples = min(len(dataset), num_examples)
                dataset = dataset.shuffle(seed=seed)
                dataset = dataset.select(range(num_examples))
        else:
            if streaming:
                dataset = HFDataset.from_generator(
                    partial(gen_from_iterable_dataset, dataset),
                    features=dataset.features
                )

        metadata = {
            "streamed": streaming,
            "seed": seed
        }

        return Dataset(
            dataset=dataset,
            text_col=text_col,
            label_col=label_col,
            is_regression=is_regression,
            metadata=metadata,
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.dataset_len
    #
    # def save_locally(self):
    #     filepath = get_dataset_filepath(dataset_name=self.dataset_name,
    #                                     split=self.split,
    #                                     max_num_examples=self.max_num_examples,
    #                                     seed=self.seed)
    #
    #     self.dataset.save_to_disk(filepath)

    def collate_fn(
            self,
            rows: dict,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 128,
            return_token_type_ids: bool = False
    ):
        texts = self._preprocess_texts(rows)
        labels = self._preprocess_labels(rows)

        tokenized = tokenizer(texts,
                              padding='max_length',
                              truncation=True,
                              max_length=max_length,
                              return_tensors='pt',
                              return_token_type_ids=return_token_type_ids)
        if return_token_type_ids:
            return tokenized.data['input_ids'], \
                   tokenized.data['attention_mask'], \
                   tokenized.data['token_type_ids'], \
                   torch.tensor(labels)

        return tokenized.data['input_ids'], tokenized.data['attention_mask'], torch.tensor(labels)

    def _preprocess_texts(self, rows):
        if isinstance(self.text_col, (list, tuple)):
            inputs = concat_columns(rows, self.text_col)
        else:
            inputs = [row[self.text_col] for row in rows]

        return inputs

    def _preprocess_labels(self, rows):
        labels = [row[self.label_col] for row in rows]
        if self.has_string_labels:
            labels = [self.class_label.str2int(label) for label in labels]

        return labels

    # def train_test_split(self, train_size=0.8, test_size=None, seed=42, shuffle=True):
    #     if test_size is not None:
    #         train_size = 1 - test_size
    #
    #     train_dataset = copy(self)
    #     test_dataset = copy(self)
    #
    #     split_internal_ds = self.dataset.train_test_split(train_size=train_size, seed=seed,
    #                                                                         shuffle=shuffle)
    #     internal_train_ds = split_internal_ds['train']
    #     internal_test_ds = split_internal_ds['test']
    #
    #     train_dataset.dataset = internal_train_ds
    #     train_dataset.dataset_len = len(internal_train_ds)
    #
    #     test_dataset.dataset = internal_test_ds
    #     test_dataset.dataset_len = len(internal_test_ds)
    #
    #     return train_dataset, test_dataset
