from torch.utils.data import SequentialSampler, DataLoader
from .LogME import LogME
import numpy as np
import torch
from .model_utils import get_pooled_output
from transformers import PreTrainedModel, PreTrainedTokenizer
from .ESM import ESM, fetch_esms
from .dataset import Dataset
from tqdm import tqdm
from typing import List
from transformers import AutoModel, AutoTokenizer


def compute_logme_esm_batch(dataset: Dataset,
                            base_model: PreTrainedModel,
                            esms: List[ESM],
                            tokenizer: PreTrainedTokenizer,
                            # regression: bool,
                            batch_size: int = 128,
                            device_name: str = "cpu"):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            collate_fn=lambda x: dataset.collate_fn(x, tokenizer=tokenizer))
    device = torch.device(device_name)
    base_model.to(device)

    regression = dataset.is_regression
    if regression:
        label_dtype = float
    else:
        label_dtype = int

    labels = np.zeros(0, label_dtype)
    esm_embeddings = [[] for _ in range(len(esms))]

    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.detach().cpu().numpy().flatten()

        with torch.no_grad():
            batch_base_embeddings = get_pooled_output(base_model, b_input_ids, b_input_mask)
            for i, transformation_net in enumerate(esms):
                batch_transformed_embeddings = transformation_net(batch_base_embeddings).cpu().numpy()
                esm_embeddings[i].append(batch_transformed_embeddings)

        labels = np.append(labels, b_labels, axis=0)

    logme_results = []
    for features in tqdm(esm_embeddings):
        embeddings = np.vstack(features)
        logme_results.append(LogME(regression=regression).fit(embeddings, labels, add_intercept=False))

    return logme_results


def compute_task_ranking(
        dataset: Dataset,
        model_name: str,
        # is_regression: bool
) -> List[tuple[str, float]]:
    esms = fetch_esms(model_name)
    repo_ids = list(esms.keys())
    esms = list(esms.values())

    bert_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = compute_logme_esm_batch(
        dataset=dataset,
        base_model=bert_model,
        tokenizer=tokenizer,
        esms=esms,
    )

    return [(esms[idx].config["task_id"], scores[idx]) for idx in np.argsort(scores)[::-1]]
