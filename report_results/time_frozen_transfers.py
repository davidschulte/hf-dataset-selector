from transfer_learning import execute_transfers
from dataset_parsing import dataset_info_dict
import os
import pickle
import time
from tqdm import tqdm

frozen_timers_dir = os.path.join('frozen_timers', 'large')
os.makedirs(frozen_timers_dir, exist_ok=True)

source_names = list(dataset_info_dict.keys())[:3]
target_names = [
    'imdb_plain_text', 'tweet_eval_emotion', 'tweet_eval_sentiment',
    'llm-book__JGLUE_JSTS', 'google_wellformed_query_default', 'paws-x_en',
    'md_gender_bias_convai2_inferred',
    'google__civil_comments_default'
]

for num_source_samples in (10000,):
    results_dict = {}

    for target in tqdm(target_names):
        start_time = time.time()

        for source in source_names[:5]:
            execute_transfers(source_dataset_name=source,
                              target_dataset_names=[target],
                              num_target_train_samples=1000,
                              num_source_samples=num_source_samples,
                              num_val_samples=1000,
                              freeze_base_model=True)

        end_time = time.time()
        results_dict[target] = (end_time - start_time) / 3 * len(dataset_info_dict)

    with open(os.path.join(frozen_timers_dir, f'{num_source_samples}.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)
