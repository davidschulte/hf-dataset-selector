import os
import pickle
from utils.path_utils import get_output_path
import time
from datetime import datetime
import json
from tqdm import tqdm
from dataset_parsing import dataset_info_dict
from hfdataset import HFDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from utils.model_utils import EMBEDDING_SIZE, create_base_model, get_pooled_output
from config import TEXTEMB_EMBEDDINGS_DIR, GPU_DEVICE, TARGET_TASKS, NUM_SOURCE_SAMPLES, NUM_TARGET_SAMPLES

overwrite_embeddings = False
stream_sources = True


def compute_text_embedding(dataset: HFDataset, batch_size=256):
    base_model = create_base_model()
    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
    base_model.to(device)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)

    embeddings_sum = np.zeros(EMBEDDING_SIZE)

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # batch_embeddings = base_model(b_input_ids, attention_mask=b_input_mask)[1]
            batch_embeddings = get_pooled_output(base_model, b_input_ids, attention_mask=b_input_mask)

        embeddings_sum += batch_embeddings.sum(axis=0).cpu().numpy()

    return embeddings_sum / len(dataset)


def main():
    source_datasets = dataset_info_dict.keys()

    for datasets_list, num_train_samples in ((TARGET_TASKS, NUM_TARGET_SAMPLES), (source_datasets, NUM_SOURCE_SAMPLES)):
        for dataset_name in tqdm(datasets_list):
            try:
                output_dir = get_output_path(TEXTEMB_EMBEDDINGS_DIR,
                                             target_name=dataset_name,
                                             num_train_samples=num_train_samples)
                os.makedirs(output_dir, exist_ok=True)

                start_time = time.time()

                dataset_embedding_filepath = os.path.join(output_dir, 'textemb.pkl')
                if os.path.isfile(dataset_embedding_filepath) and not overwrite_embeddings:
                    continue

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

            except Exception as e:
                print(dataset_name)
                print(e)


if __name__ == "__main__":
    main()
