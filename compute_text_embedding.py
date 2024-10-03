from hfdataset import HFDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from config import GPU_DEVICE
from utils.model_utils import EMBEDDING_SIZE, create_base_model, get_pooled_output


def compute_text_embedding(dataset: HFDataset, batch_size=256):
    base_model = create_base_model()
    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
    base_model.to(device)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)

    embeddings_sum = np.zeros(EMBEDDING_SIZE)

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # batch_embeddings = base_model(b_input_ids, attention_mask=b_input_mask)[1]
            batch_embeddings = get_pooled_output(base_model, b_input_ids, attention_mask=b_input_mask)

        embeddings_sum += batch_embeddings.sum(axis=0).cpu().numpy()

    return embeddings_sum / len(dataset)
