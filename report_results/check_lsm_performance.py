from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import torch
from utils.model_utils import get_pooled_output, EMBEDDING_SIZE
import os
import pandas as pd
from hfdataset import HFDataset
from tqdm import tqdm
from dataset_parsing import dataset_info_dict
from utils.path_utils import get_output_path
from utils.model_utils import create_base_model, load_sequence_classification_model_from_dir, get_base_model
from utils.load_model_utils import load_model
from math import ceil
from config import GPU_DEVICE, MODELS_SOURCES_DIR, TRANSFORMATION_NETS_FILEPATH, EVAL_BASE_DIR


target_dataset_names = ['imdb_plain_text', 'tweet_eval_emotion', 'tweet_eval_sentiment',
                        'llm-book__JGLUE_JSTS', 'google_wellformed_query_default', 'paws-x_en']
source_names = list(dataset_info_dict.keys())

num_target_samples = 1000
num_source_samples = 10000

SOURCES_BATCH_SIZE = 8

OUTPUT_BASE_DIR = os.path.join(EVAL_BASE_DIR, "lsm_performance")

def check_lsm_performance(dataset, base_model, transformation_nets, source_base_models, batch_size=128, feature_dim=EMBEDDING_SIZE):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)
    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
    base_model.to(device)
    for source_base_model in source_base_models:
        source_base_model.to(device)

    lsm_features_list = [np.zeros((0, feature_dim), float)] * len(transformation_nets)
    source_base_model_features_list = [np.zeros((0, feature_dim), float)] * len(source_base_models)

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # b_labels = b_labels.detach().cpu().numpy().flatten()

        with torch.no_grad():
            batch_base_embeddings = get_pooled_output(base_model, b_input_ids, b_input_mask)
            for i, transformation_net in enumerate(transformation_nets):
                batch_transformed_embeddings = transformation_net(batch_base_embeddings).cpu().numpy()
                lsm_features_list[i] = np.append(lsm_features_list[i], batch_transformed_embeddings, axis=0)

            for i, source_model in enumerate(source_base_models):
                source_base_model_embedding = get_pooled_output(source_model, b_input_ids, b_input_mask).cpu().numpy()
                source_base_model_features_list[i] = np.append(source_base_model_features_list[i], source_base_model_embedding, axis=0)


    return [np.mean(np.linalg.norm(lsm_features - source_model_features, axis=1)) for lsm_features, source_model_features
            in zip(lsm_features_list, source_base_model_features_list)]


def main():

    results_dict = {}

    for target_dataset_name in tqdm(target_dataset_names):
        # output_dir = os.path.join(OUTPUT_BASE_DIR, )

        target_dataset = HFDataset(target_dataset_name, split="train", max_num_examples=num_target_samples)

        base_model = create_base_model()
        target_lsm_performances = []

        num_sources = len(source_names)
        num_source_model_batches = ceil(num_sources / SOURCES_BATCH_SIZE)

        for source_model_batch_idx in tqdm(range(num_source_model_batches)):
            batch_source_names = source_names[source_model_batch_idx * SOURCES_BATCH_SIZE:min(((source_model_batch_idx + 1)*SOURCES_BATCH_SIZE), num_sources)]
            source_model_paths = [get_output_path(MODELS_SOURCES_DIR,
                                                  source_name=source_name,
                                                  num_train_samples=num_source_samples) for source_name in batch_source_names]

            source_base_models = [get_base_model(load_sequence_classification_model_from_dir(model_path))
                                  for model_path in source_model_paths]

            source_lsm_paths = [os.path.join(get_output_path(TRANSFORMATION_NETS_FILEPATH,
                                                                         num_train_samples=num_source_samples),
                                                         f'{dataset_name}.pt') for dataset_name in batch_source_names]

            lsms = [load_model(filepath,model_type='transformation_network') for filepath in source_lsm_paths]


            batch_target_lsm_performances = check_lsm_performance(target_dataset, base_model, lsms, source_base_models)
            target_lsm_performances += batch_target_lsm_performances

        results_dict[target_dataset_name] = target_lsm_performances

        for lsm_performance, source_name in zip(target_lsm_performances, source_names):
            output_dir = os.path.join(get_output_path(OUTPUT_BASE_DIR,
                                                      target_name=target_dataset_name,
                                                      source_name=source_name,
                                                      num_train_samples=num_target_samples,
                                                      num_source_samples=num_source_samples))

            os.makedirs(output_dir, exist_ok=True)

            np.save(os.path.join(output_dir, "metric.npy"), lsm_performance)

    results_df = pd.DataFrame.from_dict(results_dict).set_index(source_names)

    results_df.to_pickle(os.path.join(OUTPUT_BASE_DIR, "results.df"))


if __name__ == "__main__":
    main()
