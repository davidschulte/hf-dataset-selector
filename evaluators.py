from abc import abstractmethod
import numpy as np
import torch.nn as nn
import torch
from scipy.special import softmax
from config import GPU_DEVICE


class Evaluator:
    def __init__(self, model, metric, num_classes=None):

        self.model = model

        self.device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        self.metric = metric
        self.num_classes = num_classes

        self.labels = np.empty(0, int)
        if num_classes is None:
            self.preds = np.empty(0, int)
        else:
            self.preds = np.empty((0, num_classes), float)

    @abstractmethod
    def evaluate_step(self, *args, **kwargs):
        pass

    def reset(self):
        self.labels = np.empty(0, int)
        self.preds = np.empty(0, int)

    @property
    def evaluation_results(self):
        return self.metric.compute_results(self.labels, self.preds)


class TransformerEvaluator(Evaluator):
    def __init__(self, model, metric, num_classes=None):
        super(TransformerEvaluator, self).__init__(model, metric, num_classes)

    def evaluate_step(self, batch):
        self.model.eval()

        batch = tuple(b.to(self.device) for b in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = self.model(b_input_ids,
                                 attention_mask=b_input_mask)

        b_preds = outputs[0].detach().cpu().numpy()
        if b_preds.shape[1] > 1:
            if self.num_classes is None:
                b_preds = np.argmax(b_preds, axis=1)
            else:
                b_preds = softmax(b_preds, axis=-1)
        else:
            b_preds = b_preds.flatten()
        b_labels = b_labels.to('cpu').numpy()
        self.preds = np.append(self.preds, b_preds, axis=0)
        self.labels = np.append(self.labels, b_labels, axis=0)
