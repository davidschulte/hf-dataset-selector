from dataset_parsing import dataset_info_dict
from utils.eval_utils import get_results, save_to_latex
import pandas as pd

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

source_names_large = list(dataset_info_dict.keys())
num_target_samples = 1000
num_source_samples = 10000

k = 10

def format_ds_name(ds_name):
    ds_info = dataset_info_dict[ds_name]
    output_name = ds_info["name"].replace("__", "/")
    if ds_info["subset"] is not None:
        output_name += f":{ds_info['subset']}"

    return output_name.replace("_", "\_")


true = get_results("Transfer", target_names_large, source_names_large, num_target_samples, num_source_samples,
                   seed=None, to_pandas=True)

preds = get_results("ESM-LogME", target_names_large, source_names_large, num_target_samples, num_source_samples,
                    seed=None, to_pandas=True)
# preds = get_results("TextEmb", target_names_large, source_names_large, num_target_samples, num_source_samples, seed=None, to_pandas=True)

top_k_dfs = []
for target in target_names_large:
    top_sources = true.drop(target).sort_values(target, ascending=False)[:k].index
    top_ranked = preds.drop(target).sort_values(target, ascending=False)[:k].index

    top_source_formatted = [format_ds_name(ds_name) for ds_name in top_sources]
    top_ranked_formatted = [format_ds_name(ds_name) for ds_name in top_ranked]

    # df = pd.DataFrame.from_dict({
    #     # 'Rank': list(range(1, k + 1)),
    #     ('Optimal Ranking', 'ESM-LogME Rank'): [
    #         preds.drop(target).sort_values(target, ascending=False).index.get_loc(tr) + 1 for
    #         tr in top_sources],
    #     ('Optimal Ranking', 'Performance'): [round(100 * performance, 2) for performance in
    #                                          true.loc[top_sources, target].tolist()],
    #     ('Optimal Ranking', 'Source Task'): top_source_formatted,
    #     ('ESM-LogME Ranking', 'Source Task'): top_ranked_formatted,
    #     ('ESM-LogME Ranking', 'Performance'): [round(100 * performance, 2) for performance in
    #                                    true.loc[top_ranked, target].tolist()],
    #     ('ESM-LogME Ranking', 'True Rank'): [true.drop(target).sort_values(target, ascending=False).index.get_loc(tr) + 1 for
    #                                     tr in top_ranked]
    # })#.set_index('Rank')

    df_true = pd.DataFrame.from_dict({
        'Rank': list(range(1, k + 1)),
        'Source Task': top_source_formatted,
        'Perf.': [round(100 * performance, 2) for performance in
                                             true.loc[top_sources, target].tolist()],
        'ESM-LM Rank': [
            preds.drop(target).sort_values(target, ascending=False).index.get_loc(tr) + 1 for
            tr in top_sources],
    }).set_index('Rank')
    df_esm_logme = pd.DataFrame.from_dict({
        'Rank': list(range(1, k + 1)),
        'Source Task': top_ranked_formatted,
        'Perf.': [round(100 * performance, 2) for performance in
                                       true.loc[top_ranked, target].tolist()],
        'True Rank': [true.drop(target).sort_values(target, ascending=False).index.get_loc(tr) + 1 for
                                        tr in top_ranked]
    }).set_index('Rank')
    # df.index = df.index.astype("str")
    for df in (df_true, df_esm_logme):
        for col in df.columns:
            df[col] = df[col].astype("str")


    # df.columns = pd.MultiIndex.from_tuples(df.columns)
    save_to_latex(df_true, f'tables/top_k/true_{target}_{k}.tex', escape=False)
    save_to_latex(df_esm_logme, f'tables/top_k/esm_logme_{target}_{k}.tex', escape=False)

    # top_k_dfs.append(df)