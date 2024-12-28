from torch.utils.data import SequentialSampler, DataLoader
from .LogME import LogME
import numpy as np
import torch
from .model_utils import get_pooled_output
from transformers import PreTrainedModel, PreTrainedTokenizer
from .utils import fetch_esms, find_esm_repo_ids
from .dataset import Dataset
from .task_ranking import TaskRanking
from .ESMConfig import ESMConfig
from tqdm.auto import tqdm
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer


def compute_logme_esm_batch(dataset: Dataset,
                            base_model: PreTrainedModel,
                            esms: List["ESM"],
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

    with tqdm(dataloader, desc="Computing embeddings", unit="batch") as pbar:
        for batch in pbar:
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
    with tqdm(esm_embeddings, desc="Computing LogME", unit="Task") as pbar:
        for features in pbar:
            embeddings = np.vstack(features)
            logme_results.append(LogME(regression=regression).fit(embeddings, labels, add_intercept=False))

    return logme_results


def compute_task_ranking(
        dataset: Dataset,
        model_name: str,
        esms: Optional[List["ESM"]] = None,
        esm_repo_ids: Optional[List[str]] = None,
        # is_regression: bool
) -> TaskRanking:
# ) -> List[tuple[str, float]]:

    if esms is None:
        if esm_repo_ids is None:
           esm_repo_ids = find_esm_repo_ids(model_name=model_name)

        esms = fetch_esms(esm_repo_ids)

    bert_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = compute_logme_esm_batch(
        dataset=dataset,
        base_model=bert_model,
        tokenizer=tokenizer,
        esms=esms,
    )

    return TaskRanking([ESMConfig.from_esm(esm) for esm in esms], scores)

    # return [(esms[idx].config["task_id"], scores[idx]) for idx in np.argsort(scores)[::-1]]
