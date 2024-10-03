from torch.utils.data import Dataset as TorchDataset
from dataset_parsing import dataset_info_dict
import torch
from functools import partial
from datasets import load_dataset, load_from_disk, ClassLabel, Dataset
from config import TEXT_SEPARATOR, DATASETS_DIR
from utils.metrics import Metric
from utils import DEFAULT_TOKENIZER
import numpy as np
from copy import copy
import os


class EmptyDatasetError(Exception):
    pass


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def get_relevant_col_names(dataset_info):
    ds_input = dataset_info["input"]
    if isinstance(ds_input, str):
        ds_input = [ds_input]
    else:
        ds_input = list(ds_input)
    ds_label = dataset_info["label"]

    return ds_input + [ds_label]


def get_dataset_filepath(dataset_name, split, max_num_examples=None, seed=None):
    sample_str = f'samples_{max_num_examples}' if max_num_examples else "full"
    if seed:
        sample_str += f'_seed_{seed}'
    return os.path.join(DATASETS_DIR,
                        dataset_name,
                        split,
                        sample_str)

def load_dataset_by_info_dict(dataset_name, split, max_num_examples=None, seed=None, streaming=False, load_locally=True):
    if load_locally:
        dataset_path = get_dataset_filepath(dataset_name=dataset_name,
                                            split=split,
                                            max_num_examples=max_num_examples,
                                            seed=seed)
        if os.path.isdir(dataset_path):
            return load_from_disk(dataset_path)

    print(f"Downloading dataset {dataset_name} - {split} - {max_num_examples} - {seed}")
    dataset_info = dataset_info_dict[dataset_name]
    hf_name = dataset_info["name"]
    hf_subset = dataset_info.get("subset")
    hf_split = dataset_info["splits"][split]

    if hf_subset is None:
        dataset = load_dataset(hf_name, split=hf_split, streaming=streaming)  # [12250:12750]
    else:
        dataset = load_dataset(hf_name, hf_subset, split=hf_split, streaming=streaming)  # [12250:12750]

    dataset = dataset.select_columns(get_relevant_col_names(dataset_info))

    label_col_name = dataset_info['label']

    task_type = dataset_info.get('task_type')
    if task_type == 'classification':
        dataset = dataset.filter(lambda example: example[label_col_name] != -1)

    sample_seed = seed if seed is not None else 42
    if isinstance(max_num_examples, int):
        if streaming:
            if split == "train":
                max_size = dataset_info["train_split_size"]
                max_num_examples = min(max_num_examples, max_size)
            # self.dataset = list(self.dataset.shuffle(seed=sample_seed, buffer_size=max_num_examples).take(max_num_examples))
            dataset = dataset.shuffle(seed=sample_seed, buffer_size=100000).take(max_num_examples)
            # dataset = list(dataset)
            dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset),
                                                  features=dataset.features)
        else:
            dataset_len = len(dataset)
            num_examples = min(dataset_len, max_num_examples)
            dataset = dataset.shuffle(seed=sample_seed)
            dataset = dataset.select(range(num_examples))
    else:
        if streaming:
            dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset),
                                                  features=dataset.features)

    return dataset





def concat_columns(inputs, column_list: tuple):
    if isinstance(inputs, str):
        return TEXT_SEPARATOR.join([inputs[col] for col in column_list])

    return [TEXT_SEPARATOR.join([row[col] for col in column_list]) for row in inputs]


class HFDataset(TorchDataset):

    def __init__(self, dataset_name, split, max_num_examples=None, streaming=False, load_locally=True, seed=None):

        # if os.path.isfile(os.path.join('datasets', f'{dataset_name}.pkl')):
        #     # print('Loaded datast locally. Please take care of splits.')
        #     with open(os.path.join('datasets', f'{dataset_name}.pkl'), 'rb') as f:
        #             loaded_ds = pickle.load(f)
        #
        #     for attribute_name in loaded_ds.__dict__.keys():
        #         setattr(self,attribute_name, getattr(loaded_ds, attribute_name))
        #
        #     return

        self.dataset_name = dataset_name
        self.seed = seed
        self.max_num_examples = max_num_examples
        self.split = split

        self.dataset_info = dataset_info_dict[dataset_name]
        self.task_type = self.dataset_info.get('task_type')

        split_info = self.dataset_info['splits']
        self.dataset = load_dataset_by_info_dict(dataset_name,
                                                 split=split,
                                                 streaming=streaming,
                                                 max_num_examples=max_num_examples,
                                                 seed=seed,
                                                 load_locally=load_locally)

        self.dataset_len = len(self.dataset)

        if self.dataset_len == 0:
            raise EmptyDatasetError(f'Dataset {dataset_name} is empty.')

        label_col_name = self.dataset_info['label']
        label_features = self.dataset.features[label_col_name]

        self.has_string_labels = label_features.dtype == 'string'
        if self.has_string_labels:
            if streaming:
                label_list = sorted(list(set([example[label_col_name] for example in self.dataset])))
            else:
                label_list = sorted(list(set(self.dataset[label_col_name])))
            self.label_dim = len(label_list)
            self.class_label = ClassLabel(num_classes=self.label_dim, names=label_list)
        elif 'float' in label_features.dtype:
            self.label_dim = 1
            self.class_label = None
        else:
            try:
                self.label_dim = label_features.num_classes
            except:
                self.label_dim = np.max(self.dataset[label_col_name]) + 1
            self.class_label = None

        metric_info = self.dataset_info.get('metric')
        if not metric_info:
            metric_info = 'accuracy' if self.label_dim > 1 else ('pearson_corr', 'spearman_corr')

        self.metric = Metric(metric_info)



    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.dataset_len

    def save_locally(self):
        filepath = get_dataset_filepath(dataset_name=self.dataset_name,
                                        split=self.split,
                                        max_num_examples=self.max_num_examples,
                                        seed=self.seed)

        self.dataset.save_to_disk(filepath)

    def collate_fn(self, rows, max_length=128, return_token_type_ids=False):
        return self.preprocess(rows, DEFAULT_TOKENIZER, max_length=max_length, return_token_type_ids=return_token_type_ids)

    def preprocess_inputs(self, rows):
        input_cols = self.dataset_info['input']
        if isinstance(input_cols, tuple):
            inputs = concat_columns(rows, input_cols)
        else:
            inputs = [row[input_cols] for row in rows]

        return inputs

    def preprocess(self, rows, tokenizer, max_length=128, return_token_type_ids=False):
        inputs = self.preprocess_inputs(rows)
        label_col = self.dataset_info['label']
        labels = [row[label_col] for row in rows]
        if self.has_string_labels:
            labels = [self.class_label.str2int(label) for label in labels]

        tokenized = tokenizer(inputs,
                              padding='max_length',
                              truncation=True,
                              max_length=max_length,
                              return_tensors='pt',
                              return_token_type_ids=return_token_type_ids)
        if return_token_type_ids:
            return tokenized.data['input_ids'],\
                   tokenized.data['attention_mask'],\
                   tokenized.data['token_type_ids'],\
                   torch.tensor(labels)

        return tokenized.data['input_ids'], tokenized.data['attention_mask'], torch.tensor(labels)

    def train_test_split(self, train_size=0.8, test_size=None, seed=42, shuffle=True):
        if test_size is not None:
            train_size = 1 - test_size

        train_dataset = copy(self)
        test_dataset = copy(self)

        split_internal_ds = self.dataset.train_test_split(train_size=train_size, seed=seed,
                                                                            shuffle=shuffle)
        internal_train_ds = split_internal_ds['train']
        internal_test_ds = split_internal_ds['test']

        train_dataset.dataset = internal_train_ds
        train_dataset.dataset_len = len(internal_train_ds)

        test_dataset.dataset = internal_test_ds
        test_dataset.dataset_len = len(internal_test_ds)

        return train_dataset, test_dataset
