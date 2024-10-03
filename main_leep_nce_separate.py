import os.path
import time
import numpy as np
from datetime import datetime
from utils.path_utils import get_output_path
from utils.model_utils import load_sequence_classification_model_from_dir
from config import MODELS_SOURCES_DIR, LEEP_DIR, NCE_DIR, CLASSIFICATION_TARGETS_TASKS
from hfdataset import HFDataset
from compute_leep_and_nce import compute_leep, compute_nce
from tqdm import tqdm
import json
from dataset_parsing import dataset_info_dict

source_dataset_names = [ds_name for ds_name in dataset_info_dict.keys() if
                        dataset_info_dict[ds_name]['task_type']=='classification']

num_target_samples = 1000
num_source_samples = 10000

overwrite = False

for (prediction_method, output_base_dir) in [(compute_leep, LEEP_DIR), (compute_nce, NCE_DIR)]:
    print(f'Computing LEEP and NCE for scenario with'
          f'{num_source_samples if num_source_samples else "full"} source samples.')
    for target_dataset_name in CLASSIFICATION_TARGETS_TASKS:
        # print(f'Calculating LEEP and NCE for target dataset {target_dataset_name}')
        target_results = {}
        target_dataset = HFDataset(target_dataset_name,
                                   split='train',
                                   max_num_examples=num_target_samples)

        start_time = time.time()

        for source_dataset_name in tqdm(source_dataset_names):
            source_model_dir = get_output_path(MODELS_SOURCES_DIR,
                                               source_name=source_dataset_name,
                                               num_train_samples=num_source_samples)
            source_model = load_sequence_classification_model_from_dir(source_model_dir)
            score = prediction_method(source_model, target_dataset)
            target_results[source_dataset_name] = score
        # results_dict_leep[target_dataset_name] = target_results_leep

        time_elapsed = time.time() - start_time

        target_output_path = get_output_path(output_base_dir,
                                             num_train_samples=num_target_samples,
                                             num_source_samples=num_source_samples,
                                             target_name=target_dataset_name)

        for source_dataset_name in source_dataset_names:
            output_dir = get_output_path(target_output_path,
                                         source_name=source_dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'metric.npy'), target_results[source_dataset_name])

        timer_dict = {
            'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            'elapsed': time_elapsed,
            'num_sources': len(source_dataset_names),
        }

        with open(os.path.join(target_output_path, 'timer.json'), 'w') as f:
            json.dump(timer_dict, f)
