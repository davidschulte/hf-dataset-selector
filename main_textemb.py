import os
from hfdataset import HFDataset
from compute_text_embedding import compute_text_embedding
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
from config import TEXTEMB_EVAL_DIR, TEXTEMB_EMBEDDINGS_DIR, GPU_DEVICE, TARGET_TASKS


def similarity_score(x, y):
    return 1 - spatial.distance.cosine(x, y)


source_datasets = dataset_info_dict.keys()

num_source_samples = 10000
num_target_samples = 1000

overwrite_embeddings = False
stream_sources = True

embeddings_dict = {}

successful_sources = []

for datasets_list, num_train_samples in ((TARGET_TASKS, num_target_samples), (source_datasets, num_source_samples)):
    for dataset_name in tqdm(datasets_list):
        try:
            output_dir = get_output_path(TEXTEMB_EMBEDDINGS_DIR,
                                         target_name=dataset_name,
                                         num_train_samples=num_train_samples)
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.time()

            dataset_embedding_filepath = os.path.join(output_dir, 'textemb.pkl')
            if os.path.isfile(dataset_embedding_filepath) and not overwrite_embeddings:
                with open(dataset_embedding_filepath, 'rb') as f:
                    embedding = pickle.load(f)
            else:
                dataset = HFDataset(dataset_name,
                                    split='train',
                                    max_num_examples=num_train_samples,
                                    streaming=stream_sources)
                embedding = compute_text_embedding(dataset)

                with open(dataset_embedding_filepath, 'wb') as f:
                    pickle.dump(embedding, f)

                time_elapsed = time.time() - start_time

                device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")

                if device.type == "cuda":
                    used_device = torch.cuda.get_device_name(device)
                else:
                    used_device = "cpu"

                timer_dict = {
                    'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    'elapsed': time_elapsed,
                    'device': used_device
                }

                with open(os.path.join(output_dir, 'timer.json'), 'w') as f:
                    json.dump(timer_dict, f)

                if datasets_list == source_datasets:
                    successful_sources.append(dataset_name)
            embedding_name = f'{dataset_name}_{num_train_samples}'
            embeddings_dict[embedding_name] = embedding

        except Exception as e:
            print(dataset_name)
            print(e)

for target_dataset_name in TARGET_TASKS:
    target_results = {}

    start_time = time.time()
    target_embedding_name = f'{target_dataset_name}_{num_target_samples}'
    target_vector = embeddings_dict[target_embedding_name]
    for source_dataset_name in source_datasets:
        source_embedding_name = f'{source_dataset_name}_{num_source_samples}'
        source_vector = embeddings_dict[source_embedding_name]
        similarity = similarity_score(target_vector, source_vector)
        target_results[source_dataset_name] = similarity

    # results_dict[target_dataset_name] = target_results
    time_elapsed = time.time() - start_time

    target_output_path = get_output_path(TEXTEMB_EVAL_DIR,
                                         num_train_samples=num_target_samples,
                                         num_source_samples=num_source_samples,
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
