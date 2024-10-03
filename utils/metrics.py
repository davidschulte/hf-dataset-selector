from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, multilabel_confusion_matrix, accuracy_score, mean_squared_error
import numpy as np

base_metrics = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    # 'f1_micro': f1_score(labels, preds, average='micro'),
    # 'f1_macro': f1_score(labels, preds, average='macro'),
    # 'f1_weighted': f1_score(labels, preds, average='weighted'),
    # 'confusion_matrix': multilabel_confusion_matrix(labels, preds),
    'pearson_corr': lambda labels, preds: pearsonr(labels, preds)[0],
    'spearman_corr': lambda labels, preds: spearmanr(labels, preds)[0],
    'matthews_corr': matthews_corrcoef,
    'mse': mean_squared_error
}


class Metric:
    def __init__(self, metric_info):
        # self.task_name = task_name
        self.metric_info = metric_info

    # @staticmethod
    # def _get_main_metric_info(task_name):
    #     if not task_name:
    #         return 'accuracy'
    #
    #     main_metric_info = dataset_info_dict[task_name].get('metric')
    #     if main_metric_info is None:
    #         # TODO: Warning because there is no metric
    #         print('Main metric not found.')
    #         return 'accuracy'
    #
    #     else:
    #         return main_metric_info
        # elif callable(main_metric_info):
        #     return main_metric_info
        # elif isinstance(main_metric_info, str):
        #     return globals()[main_metric_info]
        # elif isinstance(main_metric_info, tuple):
        #     return lambda preds, labels: sum(globals()[metric](preds, labels) for metric in main_metric_info) / len(main_metric_info)

    def compute_results(self, labels, preds):
        results = {
            'labels': labels,
            'predictions': preds,
        }

        if isinstance(self.metric_info, str):
            metric_str = self.metric_info
            main_metric = base_metrics[metric_str](labels, preds)
            # results.update({'metric': base_metrics[metric_str](labels, preds), 'metric_name': metric_str})
        elif isinstance(self.metric_info, tuple):
            metric_str = ', '.join(self.metric_info)
            results.update({metric_name: base_metrics[metric_name](labels, preds) for metric_name in self.metric_info})
            main_metric = sum(results[metric_name] for metric_name in self.metric_info)/len(self.metric_info)
        elif callable(self.metric_info):
            metric_str = self.metric_info.__name__
            main_metric = self.metric_info(labels, preds)
        else:
            print('Error finding main metric.')
            metric_str, main_metric = None, None

        results['metric_info'] = metric_str
        results['metric'] = main_metric

        return results


def mean_cosine_distance(a, b):
    return (1 - (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).mean()

def mean_squared_cosine_distance(a, b):
    return ((1 - (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))**2).mean()
