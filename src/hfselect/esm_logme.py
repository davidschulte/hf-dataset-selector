from typing import Optional
from collections import defaultdict

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .logme import LogME
from .model_utils import get_pooled_output
from .utils import fetch_esms, find_esm_repo_ids
from .dataset import Dataset
from .task_ranking import TaskRanking
from .esm import ESM
from hfselect import logger


class NoESMsFoundError(Exception):
    def __init__(self):
        super().__init__("No ESMs matching the search criteria could be found.")


def compute_scores(
    dataset: Dataset,
    base_model: PreTrainedModel,
    esms: list[ESM],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 128,
    device_name: str = "cpu",
) -> list[float]:
    """
    Computes the ESM-LogME scores for all ESMs.

    Args:
        dataset: The target dataset
        base_model: The base LM used for computing embeddings
        esms: List of the ESMs representing the intermediate datasets
        tokenizer: The tokenizer used for tokenizing the target texts
        batch_size: Describes how many embeddings are computed and transformed in a batch
        device_name: The device name of the device for computation (e.g. "cpu", "cuda")

    Returns:
        scores: The ESM-LogME scores produced by the ESMs
    """
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=lambda x: dataset.collate_fn(x, tokenizer=tokenizer),
    )
    device = torch.device(device_name)
    base_model.to(device)
    for esm in esms:
        esm.to(device)

    regression = dataset.is_regression
    if regression:
        label_dtype = float
    else:
        label_dtype = int

    labels = np.zeros(0, label_dtype)
    esm_embeddings = [[] for _ in range(len(esms))]
    faulty_esm_indices = set()
    errors = defaultdict(list)

    with tqdm(dataloader, desc="Computing embeddings", unit="batch") as pbar:
        for batch in pbar:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_labels = b_labels.detach().cpu().numpy().flatten()

            with torch.no_grad():
                batch_base_embeddings = get_pooled_output(
                    base_model, b_input_ids, b_input_mask
                )
                for i, esm in enumerate(esms):
                    if i in faulty_esm_indices:
                        continue
                    try:
                        batch_transformed_embeddings = (
                            esm(batch_base_embeddings).cpu().numpy()
                        )
                        esm_embeddings[i].append(batch_transformed_embeddings)
                    except Exception as e:
                        faulty_esm_indices.add(i)
                        errors[type(e).__name__].append(esm.config.get(["repo_id"]))

            labels = np.append(labels, b_labels, axis=0)

    if len(errors) > 0:
        logger.warning(
            f"Computing embeddings failed for {len(faulty_esm_indices)} of {len(esms)} ESMs."
        )
        logger.debug(errors)

    scores = []
    with tqdm(esm_embeddings, desc="Computing LogME", unit="Task") as pbar:
        for idx, features in enumerate(pbar):
            if idx in faulty_esm_indices:
                scores.append(np.nan)
                continue

            embeddings = np.vstack(features)
            scores.append(
                LogME(regression=regression).fit(
                    embeddings, labels, add_intercept=False
                )
            )

    return scores


def compute_task_ranking(
    dataset: Dataset,
    model_name: str,
    esms: Optional[list[ESM]] = None,
    esm_repo_ids: Optional[list[str]] = None,
    batch_size: int = 128,
    device_name: str = "cpu",
) -> TaskRanking:
    """
    Computes a task ranking by first computing scores and then ranking the intermediate datasets by their scores.

    Args:
        dataset: The target dataset
        model_name: The name of the base LM used for computing embeddings
        esms: List of the ESMs representing the intermediate datasets
        esm_repo_ids: List of the HF repo IDs of the ESMs representing the intermediate datasets
        batch_size: Describes how many embeddings are computed and transformed in a batch
        device_name: The device name of the device for computation (e.g. "cpu", "cuda")

    Returns:
        task_ranking: A task ranking of the intermediate tasks. Intermediate datasets with invalid ESMS are excluded.

    """
    if esms is None:
        if esm_repo_ids is None:
            esm_repo_ids = find_esm_repo_ids(model_name=model_name)

        esms = fetch_esms(esm_repo_ids)

    if len(esms) == 0:
        logger.error(
            "No ESMs matching the search criteria could be found."
            "You can use get_esm_coverage to find out which base models have valid ESMs."
        )
        raise NoESMsFoundError

    bert_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = compute_scores(
        dataset=dataset,
        base_model=bert_model,
        tokenizer=tokenizer,
        esms=esms,
        batch_size=batch_size,
        device_name=device_name,
    )

    return TaskRanking([esm.create_config() for esm in esms], scores)
