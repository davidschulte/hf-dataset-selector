import numpy as np
from torch.utils.data import Dataset as TorchDataset
from .dataset import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import SequentialSampler, DataLoader
import os
from typing import Optional, Union, List
from tqdm import tqdm
from .model_utils import get_pooled_output
import torch


class InvalidEmbeddingDatasetError(Exception):

    def __init__(self, len_x: int, len_y: int):
        super().__init__(f"Number of base and transformed embeddings does not match: {len_x} != {len_y}.")
        self.len_x = len_x
        self.len_y = len_y


class EmbeddingDataset(TorchDataset):

    def __init__(
            self,
            x: Union[np.array, List[np.array]],
            y: Union[np.array, List[np.array]],
            ):

        if isinstance(x, list):
            x = np.vstack(x)
        if isinstance(y, list):
            y = np.vstack(y)

        if len(x) != len(y):
            raise InvalidEmbeddingDatasetError(len(x), len(y))

        self.x = x
        self.y = y

        self.num_rows = len(self.x)

    @classmethod
    def from_disk(cls, filepath):
        x = np.loadtxt(os.path.join(filepath, 'standard_embeddings.csv'))
        y = np.loadtxt(os.path.join(filepath, 'trained_embeddings.csv'))

        return EmbeddingDataset(x, y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.num_rows


def create_embedding_dataset(
        dataset: Dataset,
        base_model: PreTrainedModel,
        tuned_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device_name: str = "cpu",
        output_path: Optional[str] = None,
        batch_size: int = 128,
        overwrite: bool = True
) -> EmbeddingDataset:
    if output_path:
        standard_embeddings_filepath = os.path.join(output_path, f'standard_embeddings.csv')
        trained_embeddings_filepath = os.path.join(output_path, f'trained_embeddings.csv')
        if os.path.isfile(standard_embeddings_filepath) and os.path.isfile(
                trained_embeddings_filepath) and not overwrite:
            print("Found embeddings.")
            return EmbeddingDataset.from_disk(output_path)

        for embedding_filepath in [standard_embeddings_filepath, trained_embeddings_filepath]:
            if os.path.exists(embedding_filepath):
                os.remove(embedding_filepath)

        os.makedirs(output_path, exist_ok=True)

    device = torch.device(device_name)

    base_model.to(device)
    tuned_model.to(device)

    base_model.eval()
    tuned_model.eval()
    print('Loading models complete!')

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            collate_fn=lambda x: dataset.collate_fn(x, tokenizer=tokenizer))
    base_embeddings = []
    trained_embeddings = []

    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            base_embeddings_batch = get_pooled_output(base_model, b_input_ids,
                                                    b_input_mask).cpu().numpy()
            trained_embeddings_batch = get_pooled_output(tuned_model, b_input_ids, b_input_mask).cpu().numpy()

        if output_path:
            with open(standard_embeddings_filepath, "ab") as f:
                np.savetxt(f, base_embeddings_batch)

            with open(trained_embeddings_filepath, "ab") as f:
                np.savetxt(f, trained_embeddings_batch)

        else:
            base_embeddings.append(base_embeddings_batch)
            trained_embeddings.append(trained_embeddings_batch)

    if output_path:
        return EmbeddingDataset.from_disk(output_path)

    return EmbeddingDataset(base_embeddings, trained_embeddings)
