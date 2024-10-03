from torch.utils.data import SequentialSampler, DataLoader
from evaluators import TransformerEvaluator
from utils.LEEP import LEEP
from utils.NCE import NCE
from hfdataset import HFDataset
import numpy as np


def compute_leep_and_nce(source_model, dataset: HFDataset, batch_size=128):
    validation_sampler = SequentialSampler(dataset)
    validation_dataloader = DataLoader(dataset,
                                       sampler=validation_sampler,
                                       batch_size=batch_size,
                                       collate_fn=dataset.collate_fn)

    evaluator = TransformerEvaluator(model=source_model, metric=dataset.metric, num_classes=source_model.num_labels)

    for batch in validation_dataloader:
        evaluator.evaluate_step(batch)

    pseudo_labels = evaluator.preds
    target_labels = evaluator.labels

    leep_score = LEEP(pseudo_labels, target_labels)
    nce_score = NCE(np.argmax(pseudo_labels, axis=1), target_labels)

    return leep_score, nce_score

def compute_leep(source_model, dataset: HFDataset, batch_size=128):
    validation_sampler = SequentialSampler(dataset)
    validation_dataloader = DataLoader(dataset,
                                       sampler=validation_sampler,
                                       batch_size=batch_size,
                                       collate_fn=dataset.collate_fn)

    evaluator = TransformerEvaluator(model=source_model, metric=dataset.metric, num_classes=source_model.num_labels)

    for batch in validation_dataloader:
        evaluator.evaluate_step(batch)

    pseudo_labels = evaluator.preds
    target_labels = evaluator.labels

    leep_score = LEEP(pseudo_labels, target_labels)

    return leep_score


def compute_nce(source_model, dataset: HFDataset, batch_size=128):
    validation_sampler = SequentialSampler(dataset)
    validation_dataloader = DataLoader(dataset,
                                       sampler=validation_sampler,
                                       batch_size=batch_size,
                                       collate_fn=dataset.collate_fn)

    evaluator = TransformerEvaluator(model=source_model, metric=dataset.metric)

    for batch in validation_dataloader:
        evaluator.evaluate_step(batch)

    pseudo_labels = evaluator.preds
    target_labels = evaluator.labels

    nce_score = NCE(pseudo_labels, target_labels)

    return nce_score