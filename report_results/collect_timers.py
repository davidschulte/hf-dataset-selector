import json
import numpy as np
import pandas as pd

from utils.path_utils import get_output_path
from dataset_parsing import dataset_info_dict
from config import TASKEMB_EMBEDDINGS_DIR, TEXTEMB_EMBEDDINGS_DIR
# from config import MODELS_TRANSFER_DIR, MODELS_SOURCES_DIR, \
#     TASKEMB_FROZEN_EMBEDDINGS_DIR, TEXTEMB_EMBEDDINGS_DIR, \
#     LOGME_TNN_DIR, LOGME_DIR, LEEP_DIR, NCE_DIR, VOCAB_OVERLAP_DIR, TEXTEMB_EVAL_DIR, TASKEMB_FROZEN_EVAL_DIR

from utils.eval_utils import method_paths, save_to_latex

small = False
if small:
    num_source_samples_list = [1000, 10000, "full"]
    seed_range = range(42, 42+5)

    source_names = sorted([
    'glue_cola',
    'glue_mnli',
    'glue_mrpc',
    'glue_qnli',
    'glue_qqp',
    'glue_rte',
    'glue_sst2',
    'glue_stsb',
    'glue_wnli',
    'scitail',
    'snli'
    ])

    target_names = source_names

    regression_tasks = ["glue_stsb"]
else:
    num_source_samples_list = [10000]
    seed_range = [None]
    source_names = dataset_info_dict.keys()
    target_names = [
        'imdb_plain_text',
        'tweet_eval_emotion',
        'tweet_eval_sentiment',
        'paws-x_en',
        'md_gender_bias_convai2_inferred',
        'llm-book__JGLUE_JSTS',
        'google_wellformed_query_default',
        'google__civil_comments_default'
    ]


    regression_tasks = [ds for ds in dataset_info_dict if dataset_info_dict[ds]["task_type"] == "regression"]

num_target_samples = 1000

# for method in method_paths.keys():
methods = [m for m in method_paths if m != "Transfer"]
timers = {"Method": methods}

for num_source_samples in num_source_samples_list:
    num_source_samples_results = []
    for method in methods:
        method_target_fails = 0

        if method == "Frozen Transfer":
            num_source_samples_results.append(np.nan)
            continue

        scenario_timers = []

        method_path = method_paths[method]

        method_target_names = target_names if method not in ("LEEP", "NCE") else [ds for ds in target_names if ds not in regression_tasks]
        method_source_names = source_names if method not in ("LEEP", "NCE") else [ds for ds in source_names if ds not in regression_tasks]

        for seed in seed_range:
            # for target_dataset in target_names:
            timer_paths = [get_output_path(method_path,
                                           num_train_samples=num_target_samples,
                                           num_source_samples=num_source_samples,
                                           seed=seed,
                                           target_name=target_dataset,
                                           filename='timer.json') for target_dataset in method_target_names]

            if method in ("taskemb", "textemb"):
                embeddings_base_dir = TASKEMB_EMBEDDINGS_DIR if method == "taskemb" else TEXTEMB_EMBEDDINGS_DIR
                timer_paths += [get_output_path(embeddings_base_dir,
                                                num_train_samples=num_target_samples,
                                                seed=seed,
                                                target_name=target_dataset,
                                                filename='timer.json') for target_dataset in method_target_names]

            for timer_path in timer_paths:
                try:
                    with open(timer_path, 'r') as f:
                        scenario_timers.append(json.load(f)['elapsed'])
                except:
                    print(timer_path)
                    method_target_fails += 1

        # num_source_samples_results.append(np.mean(scenario_timers))
        num_source_samples_results.append(np.sum(scenario_timers) / (len(method_target_names)-method_target_fails) / len(method_source_names) * 1000)

    timers[f"Source Samples: {num_source_samples}"] = num_source_samples_results

df = pd.DataFrame.from_dict(timers).set_index("Method")
        # timers[(method, num_source_samples)] = np.mean(scenario_timers)
df = df.apply(lambda x: round(x,2))

df.to_csv(f"paper/tables/timers/{'small' if small else 'large'}/method_usage_timers_paper.csv")
save_to_latex(df, f"paper/tables/timers/{'small' if small else 'large'}/method_usage_timers_paper.tex")


########################################################################################################################
# EMBEDDINGS
########################################################################################################################

a=1


