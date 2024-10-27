import numpy as np
from torch.utils.data import Dataset as TorchDataset
from .dataset import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import SequentialSampler, DataLoader
import os
from typing import Optional
from tqdm import tqdm
from .model_utils import get_pooled_output
import torch


class EmbeddingDataset(TorchDataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

        assert len(x) == len(y)

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
) -> "EmbeddingDataset":
    os.makedirs(output_path, exist_ok=True)
    standard_embeddings_filepath = os.path.join(output_path, f'standard_embeddings.csv')
    trained_embeddings_filepath = os.path.join(output_path, f'trained_embeddings.csv')
    if os.path.isfile(standard_embeddings_filepath) and os.path.isfile(
            trained_embeddings_filepath) and not overwrite:
        print("Found embeddings.")
        return EmbeddingDataset.from_disk(output_path)

    for embedding_filepath in [standard_embeddings_filepath, trained_embeddings_filepath]:
        if os.path.exists(embedding_filepath):
            os.remove(embedding_filepath)

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

    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            trained_embeddings = get_pooled_output(tuned_model, b_input_ids, b_input_mask).cpu().numpy()
            standard_embeddings = get_pooled_output(base_model, b_input_ids,
                                                    b_input_mask).cpu().numpy()

        with open(standard_embeddings_filepath, "ab") as f:
            np.savetxt(f, standard_embeddings)

        with open(trained_embeddings_filepath, "ab") as f:
            np.savetxt(f, trained_embeddings)

    return EmbeddingDataset(standard_embeddings, trained_embeddings)
