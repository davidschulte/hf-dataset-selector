from typing import Optional, Union, List, Iterable
import os

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import SequentialSampler, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from .dataset import Dataset
from .model_utils import get_pooled_output
from hfselect import logger


class InvalidEmbeddingDatasetError(Exception):
    """
    This error should be raised when an embedding dataset is invalid.
    """

    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingDataset(TorchDataset):
    """
    And EmbeddingDataset contains two sets of embeddings:
    A dataset embedded using a base model and the same dataset embedded by a fine-tuned model.
    It can be used to train an ESM on the transformation of the embedding space caused by fine-tuning the model.
    """

    def __init__(
        self,
        x: Union[np.array, List[np.array]],
        y: Union[np.array, List[np.array]],
        metadata: Optional[dict] = None,
    ):
        """
        Creates an embedding dataset from two sets of embeddings

        Args:
            x: The embeddings before fine-tuning
            y: The embeddings after fine-tuning
            metadata: The metadata will be forwarded to the ESMConfig when an ESM is trained using the embeddings
        """
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
        """
        Loads an EmbeddingDataset from a local file

        Args:
            filepath: Filepath of the saved EmbeddingDataset

        Returns:
            The loaded EmbeddingDataset
        """
        embeddings = np.load(filepath, allow_pickle=True)
        x = embeddings["x"]
        y = embeddings["y"]
        if "metadata" in embeddings:
            metadata = embeddings["metadata"].item()
        else:
            metadata = None

        return EmbeddingDataset(x, y, metadata=metadata)

    def save(self, filepath: str) -> None:
        """
        Saves an EmbeddingDataset to a local file

        Args:
            filepath: Filepath to save the embedding

        Returns:

        """
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
    """
    Creates an EmbeddingDataset by embedding the same dataset with a base model and fine-tuned model

    Args:
        dataset: The dataset to be embedded
        base_model: The base model
        tuned_model: The fine-tuned model
        tokenizer: The tokenizer to be used
        device_name: The device name of the device for computation (e.g. "cpu", "cuda")
        output_path: If an output path is passed here, the EmbeddingDataset will be saved
        batch_size: The batch size for embedding the dataset

    Returns:
        The resulting EmbeddingDataset
    """
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

        metadata = {
            **{"base_model_name": base_model.config.name_or_path},
            **dataset.metadata,
        }
        embedding_dataset = EmbeddingDataset(
            base_embeddings, trained_embeddings, metadata=metadata
        )

        if output_path:
            if os.path.isfile(output_path):
                logger.warning(f"Overwriting embeddings dataset at path: {output_path}")
            embedding_dataset.save(output_path)

        return embedding_dataset
