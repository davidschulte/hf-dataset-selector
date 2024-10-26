import os
import pickle
from scipy import spatial
import numpy as np
from utils.path_utils import get_output_path
import time
from datetime import datetime
import json
from tqdm import tqdm
import torch
from dataset_parsing import dataset_info_dict
from config import TEXTEMB_EVAL_DIR, TEXTEMB_EMBEDDINGS_DIR, GPU_DEVICE, TARGET_TASKS, NUM_SOURCE_SAMPLES, \
    NUM_TARGET_SAMPLES


def similarity_score(x, y):
    return 1 - spatial.distance.cosine(x, y)


def main():
    source_datasets = dataset_info_dict.keys()

    embeddings = {}
    for dataset_names, num_samples in ((TARGET_TASKS, NUM_TARGET_SAMPLES), (source_datasets, NUM_SOURCE_SAMPLES)):
        for dataset_name in dataset_names:
            embeddings_filepath = get_output_path(TEXTEMB_EMBEDDINGS_DIR,
                                                  target_name=dataset_name,
                                                  num_train_samples=num_samples,
                                                  filename='textemb.pkl')
            with open(embeddings_filepath, 'rb') as f:
                embeddings[(dataset_name, num_samples)] = pickle.load(f)

    for target_dataset_name in TARGET_TASKS:
        target_results = {}
        start_time = time.time()
        target_vector = embeddings[(target_dataset_name, NUM_TARGET_SAMPLES)]
        for source_dataset_name in tqdm(source_datasets):
            source_vector = embeddings[(source_dataset_name, NUM_SOURCE_SAMPLES)]
            similarity = similarity_score(target_vector, source_vector)
            target_results[source_dataset_name] = similarity

        time_elapsed = time.time() - start_time

        target_output_path = get_output_path(TEXTEMB_EVAL_DIR,
                                             num_train_samples=NUM_TARGET_SAMPLES,
                                             num_source_samples=NUM_SOURCE_SAMPLES,
                                             target_name=target_dataset_name)

        for source_dataset_name in source_datasets:
            output_dir = get_output_path(target_output_path,
                                         source_name=source_dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'metric.npy'), target_results[source_dataset_name])

        device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")

        if device.type == "cuda":
            used_device = torch.cuda.get_device_name(device)
        else:
            used_device = "cpu"

        timer_dict = {
            'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            'elapsed': time_elapsed,
            'num_sources': len(source_datasets),
            'device': used_device
        }

        with open(os.path.join(target_output_path, 'timer.json'), 'w') as f:
            json.dump(timer_dict, f)


if __name__ == "__main__":
    main()
