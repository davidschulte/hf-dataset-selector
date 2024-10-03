from dataset_parsing import dataset_info_dict
from utils.eval_utils import get_results, get_method_results_overview_large, save_to_latex, get_method_results_per_scenario_large
import pandas as pd
import os

source_names_large = dataset_info_dict.keys()

classification_targets = [
    'imdb_plain_text',
    'tweet_eval_emotion',
    'tweet_eval_sentiment',
    'paws-x_en',
    'md_gender_bias_convai2_inferred'
]

regression_targets = [
    'llm-book__JGLUE_JSTS',
    'google_wellformed_query_default',
    'google__civil_comments_default'
]

target_names_large = classification_targets + regression_targets

datasets_name_mapping = {
    'imdb_plain_text': 'IMDb',
    'tweet_eval_emotion': 'TEE',
    'tweet_eval_sentiment': 'TES',
    'paws-x_en': 'PAWS-X',
    'md_gender_bias_convai2_inferred': 'MDGB',
    'llm-book__JGLUE_JSTS': 'JSTS',
    'google_wellformed_query_default': 'GWQ',
    'google__civil_comments_default': 'QCC',
    'avg': 'avg'
}


regression_datasets_large = [ds for ds in source_names_large if dataset_info_dict[ds]['task_type']=='regression']

transfer_results = get_results(method='Transfer',
                               target_names=target_names_large,
                              source_names=source_names_large,
                              num_target_samples=1000,
                               num_source_samples=10000,
                              seed=None,
                              to_pandas=True)


methods_large = [
    'ESM-LogME',
    'LogME',
    'Vocabulary Overlap',
    'TaskEmb',
    'TextEmb',
    'Frozen Transfer',
    'NCE',
    'LEEP',
]

metrics = [
    # 'TROR',
    # 'ROBS',
    'NDCG',
    # 'ir',
    'Regret@1',
    'Regret@3',
    'Regret@5'
]


################################################################################################################
dfs = []
for prefix, target_names in (('Classification', classification_targets),
                               ('Regression', regression_targets)):

    df = get_method_results_overview_large(target_names=target_names,
                                           source_names=source_names_large,
                                           metrics=metrics,
                                           include_stds=False,
                                           regression_datasets=regression_datasets_large,
                                           methods=methods_large)

    df = df.rename(columns={c: (prefix, c) for c in df.columns})
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    dfs.append(df)

results_df = pd.concat(dfs, axis=1)


save_to_latex(results_df, os.path.join('paper', 'tables', 'method_evals', 'large', 'per_target_task_type', 'overview.tex'))

for method in methods_large:

    source_names = source_names_large
    target_names = target_names_large

    if method in ('LEEP', 'NCE'):
        exclude_datasets = regression_datasets_large
    else:
        exclude_datasets = None

    df = get_method_results_per_scenario_large(method_name=method,
                                         target_names=target_names,
                                         source_names=source_names,
                                         metrics=metrics,
                                         include_stds=False,
                                           excluded_datasets=exclude_datasets)

    df.index = df.index.map(datasets_name_mapping)
    save_to_latex(df, os.path.join('paper', 'tables', 'method_evals', 'large', 'per_method', f'{method}.tex'))