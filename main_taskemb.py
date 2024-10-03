import numpy as np
import os
from utils.path_utils import get_output_path
from scipy import spatial
from scipy.stats import rankdata
from tqdm import tqdm
import time
from datetime import datetime
import json
from dataset_parsing import dataset_info_dict
from config import TASKEMB_EMBEDDINGS_DIR, TASKEMB_EVAL_DIR, TARGET_TASKS


num_source_samples = 10000
num_target_samples = 1000
use_only_cls = False

source_names = dataset_info_dict.keys()

def compute_cosine_distances(target_embeddings_dir, source_embeddings_dirs, use_only_cls=False):
    if isinstance(source_embeddings_dirs, str):
        source_embeddings_dirs = [source_embeddings_dirs]

    if use_only_cls:
        filenames = ["cls_output.npy"]
    else:
        filenames = [filename for filename in os.listdir(target_embeddings_dir) if not filename.startswith('classifier')
                     and not filename.endswith('.json')]
    target_distances = []

    for source_embeddings_dir in source_embeddings_dirs:
        distances = np.zeros(len(filenames))

        for i, filename in enumerate(filenames):
            target_emb = np.load(os.path.join(target_embeddings_dir, filename)).flatten()
            source_emb = np.load(os.path.join(source_embeddings_dir, filename)).flatten()

            cosine_distance = spatial.distance.cosine(target_emb, source_emb)

            distances[i] = cosine_distance

        target_distances.append(distances)

    return np.vstack(target_distances)


def rank_by_component(target_distances):
    return rankdata(target_distances, axis=0)


def reciprocal_rank_fusion(target_distance_rankings):
    rankings_transformed = 1 / (60 + target_distance_rankings)
    return rankings_transformed.sum(axis=1).flatten()


def rank_sources(target_distances):
    target_distance_rankings = rank_by_component(target_distances)
    fused_rankings = reciprocal_rank_fusion(target_distance_rankings)
    return fused_rankings


if __name__ == '__main__':
    for target_dataset_name in tqdm(TARGET_TASKS):
        start_time = time.time()

        # target_distances = None

        target_embeddings_dir = get_output_path(TASKEMB_EMBEDDINGS_DIR,
                                                target_name=target_dataset_name,
                                                num_train_samples=num_target_samples)

        source_embeddings_dirs = [get_output_path(TASKEMB_EMBEDDINGS_DIR,
                                                target_name=source_name,
                                                num_train_samples=num_source_samples)
                                  for source_name in source_names]

        target_distances = compute_cosine_distances(target_embeddings_dir,
                                                    source_embeddings_dirs,
                                                    use_only_cls=use_only_cls)

        target_results = rank_sources(target_distances)

        time_elapsed = time.time() - start_time

        target_output_path = get_output_path(TASKEMB_EVAL_DIR,
                                             num_train_samples=num_target_samples,
                                             num_source_samples=num_source_samples,
                                             target_name=target_dataset_name)

        for source_name, result in zip(source_names, target_results):
            output_dir = get_output_path(target_output_path,
                                         source_name=source_name)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'metric.npy'), result)

        timer_dict = {
            'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            'elapsed': time_elapsed,
            'num_sources': len(source_names)
        }

        with open(os.path.join(target_output_path, 'timer.json'), 'w') as f:
            json.dump(timer_dict, f)

