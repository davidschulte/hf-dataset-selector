import pickle

from preparation_phase.compute_vocabulary_overlap import get_vocabulary_set, compute_jaccard_index
from hfdataset import HFDataset
from dataset_parsing import dataset_info_dict
from utils.path_utils import get_output_path
import os
import time
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
from config import VOCAB_OVERLAP_DIR, TARGET_TASKS, NUM_SOURCE_SAMPLES, NUM_TARGET_SAMPLES

source_dataset_names = list(dataset_info_dict.keys())

delete_cache = False
stream_datasets = True


failed_sources = []

for target_dataset_name in TARGET_TASKS:
    target_output_dir = get_output_path(VOCAB_OVERLAP_DIR,
                                        num_train_samples=NUM_TARGET_SAMPLES,
                                        num_source_samples=NUM_SOURCE_SAMPLES,
                                        target_name=target_dataset_name)


    start_time = time.time()

    target_dataset = HFDataset(target_dataset_name,
                               split='train',
                               max_num_examples=NUM_TARGET_SAMPLES,
                               streaming=stream_datasets)

    target_vocabulary_set = get_vocabulary_set(target_dataset)

    for source_dataset_name in tqdm(source_dataset_names):

        output_dir = get_output_path(target_output_dir, source_name=source_dataset_name)
        if os.path.isfile(os.path.join(output_dir, 'metric.npy')):
            continue

        try:
            source_dataset = HFDataset(source_dataset_name,
                                       split='train',
                                       max_num_examples=NUM_SOURCE_SAMPLES,
                                       streaming=stream_datasets)

            source_vocabulary_set = get_vocabulary_set(source_dataset)

            vocabulary_overlap = compute_jaccard_index(target_vocabulary_set, source_vocabulary_set)
            # overlap_score = compute_vocabulary_overlap(target_dataset, source_dataset)
            # print(f'Overlap of {target_dataset_name} and {source_dataset_name}: {round(overlap_score, 2)}')
            # target_results.append(vocabulary_overlap)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'metric.npy'), vocabulary_overlap)
        except Exception as e:
            print(source_dataset_name)
            print(str(e))
            failed_sources.append(source_dataset_name)

    time_elapsed = time.time() - start_time

    # results_dict[target_dataset_name] = target_results

    timer_dict = {
        'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'elapsed': time_elapsed,
        'num_sources': len(source_dataset_names)
    }

    with open(os.path.join(target_output_dir, 'timer.json'), 'w') as f:
        json.dump(timer_dict, f)

with open(os.path.join(VOCAB_OVERLAP_DIR, 'failed_sources.pkl'), 'wb') as f:
    pickle.dump(failed_sources, f)
