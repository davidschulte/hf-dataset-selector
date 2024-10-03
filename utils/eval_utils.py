import pickle
import os
import pandas as pd
import math
import numpy as np
from scipy.stats import spearmanr
from utils.path_utils import get_output_path

from config import EVAL_TRANSFER_DIR, EVAL_FROZEN_TRANSFER_DIR,  TASKEMB_EVAL_DIR, VOCAB_OVERLAP_DIR, TEXTEMB_EVAL_DIR,\
    LEEP_DIR, NCE_DIR, LOGME_DIR, LOGME_TNN_DIR, TASKEMB_FROZEN_EVAL_DIR, EVAL_BASE_DIR

#%%

method_paths = {
    'Transfer': EVAL_TRANSFER_DIR,
    'ESM-LogME': LOGME_TNN_DIR,
    'LogME': LOGME_DIR,
    'Vocabulary Overlap': VOCAB_OVERLAP_DIR,
    'TaskEmb': TASKEMB_EVAL_DIR,
    'TextEmb': TEXTEMB_EVAL_DIR,
    'Frozen Transfer': EVAL_FROZEN_TRANSFER_DIR,
    'NCE': NCE_DIR,
    'LEEP': LEEP_DIR,
}

metric_roundings = {
    'ROBS': 1,
    'TROR': 1,
    'NDCG': 0,
    'spearman': 2,
    'ir': 4,
    'Regret': 2
}


def round_result(result, rounding):
    if isinstance(result, list) or isinstance(result, tuple):
        return [round_result(r, rounding) for r in result]

    if math.isnan(result):
        return result

    rounded = round(result, rounding)
    if rounding == 0:
        rounded = int(rounded)

    return rounded


def results_to_pandas(results, source_names):
    for target_name in results.keys():
        results[target_name] = list(results[target_name].values())
    results['Sources'] = source_names

    return pd.DataFrame.from_dict(results).set_index('Sources')


def get_results(method, target_names, source_names, num_target_samples, num_source_samples, seed=None, to_pandas=False):
    # if method == 'taskemb':
    #     return pd.read_pickle()

    main_path = get_output_path(method_paths[method], num_train_samples=num_target_samples,
                                num_source_samples=num_source_samples, seed=seed)

    results = {}
    for target_name in target_names:
        target_path = os.path.join(main_path, target_name)
        target_results = {}
        for source_name in source_names:
            try:
                if 'Transfer' in method:

                    with open(os.path.join(target_path, f'{source_name}.pkl'), 'rb') as f:
                        performance_results = pickle.load(f)

                    performance_metric = performance_results.get('metric')
                    # if performance_results["metric_info"] != "accuracy":
                    #     performance_metric = (-1) * mean_squared_error(performance_results["labels"], performance_results["predictions"])

                    target_results[source_name] = performance_metric

                else:
                    metric_path = os.path.join(target_path, source_name, 'metric.npy')
                    target_results[source_name] = np.load(metric_path).item()

            except Exception as e:
                print(e)
                print(f'Could not find {method} {source_name} --> {target_name}.')
                target_results[source_name] = np.nan

        results[target_name] = target_results

    if to_pandas:
        return results_to_pandas(results, source_names)

    return results


def get_avg_results(method, target_names, source_names, num_target_samples, num_source_samples,
                    seed_range=range(42, 42 + 5), to_pandas=False):
    seed_results = [get_results(method, target_names, source_names, num_target_samples, num_source_samples, seed=seed)
                    for seed in seed_range]

    avg_results = {}
    for target_name in target_names:
        avg_target_results = {}
        for source_name in source_names:
            avg_target_results[source_name] = np.mean(
                [seed_result[target_name][source_name] for seed_result in seed_results])

        avg_results[target_name] = avg_target_results

    if to_pandas:
        return results_to_pandas(avg_results, source_names)

    return avg_results


def get_ranking(df, target):
    return df[target].sort_values(na_position='first').index[::-1]


def get_best_source(df, target):
    # return df[target].idxmax()
    return df[target].drop(target).idxmax()


def get_rank_of_source(df, source, target):
    df = df.copy()
    df.loc[target, target] = np.NINF

    score = df.loc[source, target]
    rank_without_duplicate_scores = (df[target] > score).sum() + 1
    rank_with_duplicate_scores = rank_without_duplicate_scores + (sum(df[target] == score) - 1) / 2
    return rank_with_duplicate_scores


def compute_DCG(ranking, ground_truth_values):
    dcg = 0
    for i, source in enumerate(ranking, start=1):
        dcg += (2 ** (ground_truth_values.loc[source] * 100) - 1) / np.log2(i + 1)

    return dcg


def compute_NDCG(target, true_ranking, preds_ranking, true):
    dcg_true = compute_DCG(true_ranking, true[target])
    dcg_preds = compute_DCG(preds_ranking, true[target])

    return dcg_preds / dcg_true


def get_NDCG(true, preds, target):
    preds_ranking = get_ranking(preds, target).drop(target)
    true_ranking = get_ranking(true, target).drop(target)

    return compute_NDCG(target, true_ranking, preds_ranking, true)


def get_ROBS(true, preds, target):
    best_source = get_best_source(true, target)

    return get_rank_of_source(preds, best_source, target)


def get_TROR(true, preds, target):
    best_ranked = get_best_source(preds, target)

    return get_rank_of_source(true, best_ranked, target)


def get_IR(true, preds, target):

    base_performance = true.loc[target, target]

    best_source = get_best_source(true, target)
    best_performance = true.loc[best_source, target]

    best_ranked = get_best_source(preds, target)
    pred_performance = true.loc[best_ranked, target]

    return (pred_performance - base_performance) / (best_performance - base_performance)

def get_spearman(true, preds, target):
    return spearmanr(true[target].to_numpy(), preds[target].to_numpy())[0]

def get_regret(true, preds, target, k=1):
    best_performance = true[target].max()
    top_ranked_k = preds[target].drop(target).sort_values(ascending=False)[:k].index
    top_ranked_k_performance = true.loc[top_ranked_k, target].max()

    return (best_performance - top_ranked_k_performance) / best_performance

def evaluate_method(trues, predss, metric, exclude_datasets=None, per_target=True):
    if isinstance(predss, pd.DataFrame):
        predss = [predss]


    if isinstance(trues, pd.DataFrame):
        trues = [trues for _ in range(len(predss))]


    targets = trues[0].columns

    if exclude_datasets:
        for ds_name in exclude_datasets:
            # trues = [true.drop(ds_name, axis=1).drop(ds_name) for true in trues]
            trues = [true.drop(ds_name) for true in trues]
            if ds_name in trues[0].columns:
                trues = [true.drop(ds_name, axis=1) for true in trues]

            # TODO: Check if handling is correct
            # predss = [preds.drop(ds_name) for preds in predss]
    else:
        exclude_datasets = []

    target_means = []
    target_stds = []

    for target in targets:
        if target in exclude_datasets:
            target_means.append(np.nan)
            target_stds.append(np.nan)
            continue

        seed_results = []
        for true, preds in zip(trues, predss):
            if metric == 'ROBS':
                result = get_ROBS(true, preds, target)
            elif metric == 'TROR':
                result = get_TROR(true, preds, target)
            elif metric == 'NDCG':
                result = get_NDCG(true, preds, target) * 100
            elif metric == 'spearman':
                result = get_spearman(true, preds, target)
            elif metric == 'ir':
                result = get_IR(true, preds, target)
            elif metric.startswith('Regret@'):
                result = get_regret(true, preds, target, k=int(metric.split('Regret@')[1])) * 100
            else:
                raise Exception('Unknown evaluation metric')
            seed_results.append(result)

        target_means.append(np.mean(seed_results))
        target_stds.append(np.std(seed_results))

    target_means.append(np.nanmean(target_means))
    target_stds.append(np.nanmean(target_stds))

    if per_target:
        return target_means, target_stds

    return target_means[-1], target_stds[-1]


def get_method_results_overview(target_names, source_names, num_target_samples, num_source_samples_list, metrics,
                                methods=None, seed_range=range(42, 42 + 5), include_stds=False,
                                regression_datasets=None):
    if not regression_datasets:
        regression_datasets = []

    if methods is None:
        methods = [method for method in method_paths.keys() if method != 'Transfer']

    df_dict = {}

    for num_source_samples in num_source_samples_list:
        if num_source_samples == 1000:
            scenario_str = '1000 Sources samples'
        elif num_source_samples == 10000:
            scenario_str = '10000 Source samples'
        elif num_source_samples == 'full':
            scenario_str = 'Unlimited Source samples'
        else:
            scenario_str = 'Unknown Scenario'

        trues = [get_results(method='Transfer',
                             target_names=target_names,
                             source_names=source_names,
                             num_target_samples=num_target_samples,
                             num_source_samples=num_source_samples,
                             seed=seed,
                             to_pandas=True) for seed in seed_range]

        for metric in metrics:
            metric_rounding = metric_roundings[metric]

            method_preds = {method_name: [get_results(method=method_name,
                                                      target_names=target_names if method_name not in ('LEEP', 'NCE')
                                                      else [ds for ds in target_names if ds not in regression_datasets],
                                                      source_names=source_names if method_name not in ('LEEP', 'NCE')
                                                      else [ds for ds in source_names if ds not in regression_datasets],
                                                      num_target_samples=num_target_samples,
                                                      num_source_samples=num_source_samples,
                                                      seed=seed,
                                                      to_pandas=True) for seed in seed_range] for method_name in
                            methods}

            method_results = []
            for method, preds in method_preds.items():

                if method in ['LEEP', 'NCE']:
                    exclude_datasets = regression_datasets
                else:
                    exclude_datasets = None

                method_results.append(round_result(
                    evaluate_method(trues, preds, metric=metric, exclude_datasets=exclude_datasets,
                                    per_target=False)[0], metric_rounding))

            df_dict[(scenario_str, metric)] = method_results

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=methods).T

    if include_stds:
        df = df.applymap(lambda x: '-' if 'nan' in x else x)
    else:
        df = df.fillna('-')

    return df

def get_method_results_overview_small(target_names, source_names, num_target_samples, num_source_samples, metrics,
                                methods=None, seed_range=range(42, 42 + 5), include_stds=False,
                                regression_datasets=None):
    if not regression_datasets:
        regression_datasets = []

    if methods is None:
        methods = [method for method in method_paths.keys() if method != 'Transfer']

    df_dict = {}

    if num_source_samples == 1000:
        scenario_str = '1000 Sources samples'
    elif num_source_samples == 10000:
        scenario_str = '10000 Source samples'
    elif num_source_samples == 'full':
        scenario_str = 'Unlimited Source samples'
    else:
        scenario_str = 'Unknown Scenario'

    trues = [get_results(method='Transfer',
                         target_names=target_names,
                         source_names=source_names,
                         num_target_samples=num_target_samples,
                         num_source_samples=num_source_samples,
                         seed=seed,
                         to_pandas=True) for seed in seed_range]

    for metric in metrics:
        metric_rounding = metric_roundings[metric]

        method_preds = {method_name: [get_results(method=method_name,
                                                  target_names=target_names if method_name not in ('LEEP', 'NCE')
                                                  else [ds for ds in target_names if ds not in regression_datasets],
                                                  source_names=source_names if method_name not in ('LEEP', 'NCE')
                                                  else [ds for ds in source_names if ds not in regression_datasets],
                                                  num_target_samples=num_target_samples,
                                                  num_source_samples=num_source_samples,
                                                  seed=seed,
                                                  to_pandas=True) for seed in seed_range] for method_name in
                        methods}

        method_results = []
        for method, preds in method_preds.items():

            if method in ['LEEP', 'NCE']:
                exclude_datasets = regression_datasets
            else:
                exclude_datasets = None

            method_eval_result = evaluate_method(trues, preds, metric=metric, exclude_datasets=exclude_datasets,
                                per_target=False)
            if not include_stds:
                method_eval_result = [0]
            method_results.append(round_result(method_eval_result, metric_rounding))

        df_dict[metric] = method_results

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=methods).T

    if include_stds:
        df = df.applymap(lambda x: '-' if 'nan' in x else x)
    else:
        df = df.fillna('-')

    return df

def get_method_results_overview_large(target_names, source_names, metrics,
                                      num_target_samples=1000, num_source_samples=10000,
                                methods=None, include_stds=False,
                                regression_datasets=None):
    if not regression_datasets:
        regression_datasets = []

    if methods is None:
        methods = [method for method in method_paths.keys() if method != 'Transfer']

    df_dict = {}



    true = get_results(method='Transfer',
                         target_names=target_names,
                         source_names=source_names,
                         num_target_samples=num_target_samples,
                         num_source_samples=num_source_samples,
                         to_pandas=True)

    for metric in metrics:
        metric_rounding = metric_roundings[metric if not metric.startswith("Regret") else metric.split("@")[0]]

        method_preds = {method_name: get_results(method=method_name,
                                                  target_names=target_names if method_name not in ('LEEP', 'NCE')
                                                  else [ds for ds in target_names if ds not in regression_datasets],
                                                  source_names=source_names if method_name not in ('LEEP', 'NCE')
                                                  else [ds for ds in source_names if ds not in regression_datasets],
                                                  num_target_samples=num_target_samples,
                                                  num_source_samples=num_source_samples,
                                                  to_pandas=True) for method_name in methods}

        method_results = []
        for method, preds in method_preds.items():

            if method in ['LEEP', 'NCE']:
                exclude_datasets = regression_datasets
            else:
                exclude_datasets = None

            method_results.append(round_result(
                evaluate_method(true, preds, metric=metric, exclude_datasets=exclude_datasets,
                                per_target=False)[0], metric_rounding))

        df_dict[metric] = method_results

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=methods).T

    if include_stds:
        df = df.applymap(lambda x: '-' if 'nan' in x else x)
    else:
        df = df.fillna('-')

    return df



def get_method_results_per_scenario(method_name, target_names, source_names, num_target_samples_list,
                                    num_source_samples_list, metrics, include_stds=False, excluded_datasets=None,
                                    seed_range=range(42, 42 + 5)):
    if not excluded_datasets:
        excluded_datasets = []

    df_dict = {}

    for num_source_samples in num_source_samples_list:
        if num_source_samples == 1000:
            scenario_str = '1000 Sources samples'
        elif num_source_samples == 10000:
            scenario_str = '10000 Source samples'
        elif num_source_samples == 'full':
            scenario_str = 'Unlimited Source samples'
        else:
            scenario_str = 'Unknown Scenario'

        for num_target_samples in num_target_samples_list:

            trues = [
                get_results('transfer', target_names, source_names, num_target_samples, num_source_samples, seed=seed,
                            to_pandas=True) for seed in seed_range]


            preds = [
                get_results(method_name, target_names, source_names, num_target_samples, num_source_samples, seed=seed,
                            to_pandas=True) for seed in seed_range]

        for metric in metrics:

            results_means, results_stds = evaluate_method(trues, preds, metric, exclude_datasets=excluded_datasets)

            metric_rounding = metric_roundings[metric]
            if include_stds:
                df_dict[(scenario_str, metric)] = [
                    f'{round_result(results_mean, metric_rounding)} $\pm$ {round_result(results_std, metric_rounding)}'
                    for results_mean, results_std
                    in zip(results_means, results_stds)]
            else:
                df_dict[(scenario_str, metric)] = round_result(results_means, metric_rounding)

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=target_names + ['avg']).T

    if include_stds:
        df = df.applymap(lambda x: '-' if 'nan' in x else x)
    else:
        df = df.fillna('-')

    return df


def get_method_results_per_scenario_large(method_name, target_names, source_names, metrics, num_target_samples=1000,
                                    num_source_samples=10000, include_stds=False, excluded_datasets=None,):
    if not excluded_datasets:
        excluded_datasets = []

    df_dict = {}

    true = get_results('Transfer', target_names, source_names, num_target_samples, num_source_samples,
                    to_pandas=True)

    if method_name in ('LEEP', 'NCE'):
        method_source_names = [ds for ds in source_names if ds not in excluded_datasets]
        method_target_names = [ds for ds in target_names if ds not in excluded_datasets]
    else:
        method_source_names = source_names
        method_target_names = target_names

    preds = get_results(method_name, method_target_names, method_source_names, num_target_samples, num_source_samples,
                    to_pandas=True)

    for metric in metrics:

        results_means, results_stds = evaluate_method(true, preds, metric, exclude_datasets=excluded_datasets)

        metric_rounding = metric_roundings[metric if not metric.startswith("Regret") else metric.split("@")[0]]
        if include_stds:
            df_dict[metric] = [
                f'{round_result(results_mean, metric_rounding)} $\pm$ {round_result(results_std, metric_rounding)}'
                for results_mean, results_std
                in zip(results_means, results_stds)]
        else:
            df_dict[metric] = round_result(results_means, metric_rounding)

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=target_names + ['avg']).T

    if include_stds:
        df = df.applymap(lambda x: '-' if 'nan' in x else x)
    else:
        df = df.fillna('-')

    return df

def get_method_results_per_scenario_small(method_name, target_names, source_names, metrics, num_target_samples=1000,
                                          num_source_samples_list=[1000, 10000, "full"], seed_range=range(42, 42+5),
                                          include_stds=False, excluded_datasets=None):
    if not excluded_datasets:
        excluded_datasets = []

    df_dict = {}

    for metric in metrics:
        df_dict[metric] = []

    scenario_strs = []

    for num_source_samples in num_source_samples_list:
        if num_source_samples == 1000:
            scenario_str = '1000 Sources samples'
        elif num_source_samples == 10000:
            scenario_str = '10000 Source samples'
        elif num_source_samples == 'full':
            scenario_str = 'Unlimited Source samples'
        else:
            scenario_str = 'Unknown Scenario'

        scenario_strs.append(scenario_str)

        true = [get_results('Transfer', target_names, source_names, num_target_samples, num_source_samples, seed=seed,
                        to_pandas=True) for seed in seed_range]

        if method_name in ('LEEP', 'NCE'):
            method_source_names = [ds for ds in source_names if ds not in excluded_datasets]
            method_target_names = [ds for ds in target_names if ds not in excluded_datasets]
        else:
            method_source_names = source_names
            method_target_names = target_names

        preds = [get_results(method_name, method_target_names, method_source_names, num_target_samples, num_source_samples,
                             seed=seed, to_pandas=True) for seed in seed_range]

        for metric in metrics:

            results_means, results_stds = evaluate_method(true, preds, metric, exclude_datasets=excluded_datasets,
                                                          per_target=False)

            metric_rounding = metric_roundings[metric]
            if include_stds:
                # df_dict[metric] = [
                #     f'{round_result(results_mean, metric_rounding)} $\pm$ {round_result(results_std, metric_rounding)}'
                #     for results_mean, results_std
                #     in zip(results_means, results_stds)]
                df_dict[metric].append(f'{round_result(results_means, metric_rounding)} $\pm$ {round_result(results_stds, metric_rounding)}')
            else:
                df_dict[metric].append(round_result(results_means, metric_rounding))

    # df = pd.DataFrame.from_dict(df_dict, orient='index', columns=target_names + ['avg']).T
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=scenario_strs).T

    if include_stds:
        df = df.applymap(lambda x: '-' if 'nan' in x else x)
    else:
        df = df.fillna('-')

    return df

def get_results_per_metric(method_names, target_names, source_names, num_target_samples,
                                    num_source_samples, metric, regression_datasets=None, seed_range=[None]):
    if regression_datasets is None:
        regression_datasets = []

    metric_rounding = metric_roundings[metric if not metric.startswith("Regret") else metric.split("@")[0]]

    trues = [get_results('Transfer', target_names, source_names, num_target_samples, num_source_samples,
                         seed=seed, to_pandas=True) for seed in seed_range]

    df_dict = {}

    for method_name in method_names:
        method_results = []

        method_source_names = [source for source in source_names if source not in regression_datasets] if method_name in ('LEEP', 'NCE') else source_names
        # method_source_names = source_names
        method_target_names = [target for target in target_names if target not in regression_datasets] if method_name in ('LEEP', 'NCE') else target_names
        method_preds = [get_results(method_name, method_target_names, method_source_names, num_target_samples, num_source_samples,
                                    seed=seed, to_pandas=True) for seed in seed_range]

        excluded_datasets = regression_datasets if method_name in ('LEEP', 'NCE') else []
        results_means, results_stds = evaluate_method(trues, method_preds, metric, exclude_datasets=excluded_datasets)


        df_dict[method_name] = round_result(results_means, metric_rounding)

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=target_names + ['avg']).T

    return df


def format_for_output(cell):
    if isinstance(cell, list):
        return " \pm ".join([str(x_) for x_ in cell])
    if isinstance(cell, float):
        if cell.is_integer():
            cell = int(cell)

    return str(cell)

def save_to_latex(df, output_path, escape=True, multicolum=None):
    # df.index = df.index.str.replace('glue_', '')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # for col in df.columns:
    #     if df[col].apply(float.is_integer).all():

    # df = df.applymap(lambda x: " \pm ".join([str(x_) for x_ in x]) if isinstance(x, list) else str(x))
    df = df.applymap(format_for_output)
    # if "NDCG" in df.columns:
    #     df["NDCG"] = df["NDCG"].astype(str)

    # df.index = [index if '__' not in index else index.split('__')[1] for index in df.index]

    if escape:
        # styled = df.style.format(precision=2).format_index(escape="latex", axis=1).format_index(escape="latex", axis=0).to_latex(hrules=True)
        styled = df.style.format_index(escape="latex", axis=1).format_index(escape="latex", axis=0).to_latex(hrules=True)
    else:
        styled = df.style.to_latex(hrules=True)


    with open(output_path, 'w') as f:
        # f.write(df.style.format(precision=2).to_latex())
        f.write(styled)
