import numpy as np
from torch.utils.data import Dataset as TorchDataset
from .dataset import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import SequentialSampler, DataLoader
import os
from typing import Optional, Union, List, Iterable
from tqdm import tqdm
from .model_utils import get_pooled_output
import torch
import warnings


class InvalidEmbeddingDatasetError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingDataset(TorchDataset):
    def __init__(
        self, x: Union[np.array, List[np.array]], y: Union[np.array, List[np.array]], metadata: Optional[dict] = None
    ):
        if isinstance(x, list):
            x = np.vstack(x)
        if isinstance(y, list):
            y = np.vstack(y)

        if len(x) != len(y):
            raise InvalidEmbeddingDatasetError(
                f"Number of base and transformed embeddings does not match: {len(x)} != {len(y)}."
            )

        if x.shape[1] != y.shape[1]:
            raise InvalidEmbeddingDatasetError(
                f"Dimension of base and transformed embeddings does not match: {x.shape[1]} != {y.shape[1]}."
            )

        self.x = x
        self.y = y
        self.metadata = metadata or {}
        self.embedding_dim = x.shape[1]
        self.num_rows = len(self.x)

    @classmethod
    def from_disk(cls, filepath: str):
        embeddings = np.load(filepath)
        embeddings = np.load(filepath, allow_pickle=True)
        x = embeddings["x"]
        y = embeddings["y"]
        if "metadata" in embeddings:
            metadata = embeddings["metadata"].item()
        else:
            metadata = None

        return EmbeddingDataset(x, y)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, x=self.x, y=self.y, metadata=np.array(self.metadata))

    def __getitem__(self, idx: Union[int, Iterable[int]]):
        if isinstance(idx, int):
            return self.x[idx], self.y[idx]
            # return EmbeddingDataset(self.x[idx][None, :], self.y[idx][None, :])

        return EmbeddingDataset(self.x[idx], self.y[idx])

    def __len__(self) -> int:
        return self.num_rows


def create_embedding_dataset(
    dataset: Dataset,
    base_model: PreTrainedModel,
    tuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device_name: str = "cpu",
    output_path: Optional[str] = None,
    batch_size: int = 128,
) -> EmbeddingDataset:
    device = torch.device(device_name)

    base_model.to(device)
    tuned_model.to(device)

    base_model.eval()
    tuned_model.eval()

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=lambda x: dataset.collate_fn(x, tokenizer=tokenizer),
    )
    base_embeddings = []
    trained_embeddings = []

    with tqdm(dataloader, desc="Computing embedding dataset", unit="batch") as pbar:
        for batch in pbar:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, _ = batch

            with torch.no_grad():
                base_embeddings_batch = (
                    get_pooled_output(base_model, b_input_ids, b_input_mask)
                    .cpu()
                    .numpy()
                )
                trained_embeddings_batch = (
                    get_pooled_output(tuned_model, b_input_ids, b_input_mask)
                    .cpu()
                    .numpy()
                )

            base_embeddings.append(base_embeddings_batch)
            trained_embeddings.append(trained_embeddings_batch)

        metadata = {**{"base_model_name": base_model.config.name_or_path}, **dataset.metadata}
        embedding_dataset = EmbeddingDataset(base_embeddings, trained_embeddings, metadata=metadata)

        if output_path:
            if os.path.isfile(output_path):
                warnings.warn(f"Overwriting embeddings dataset at path: {output_path}")
            embedding_dataset.save(output_path)

        return embedding_dataset
