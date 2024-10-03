from torch.utils.data import DataLoader, SequentialSampler
from hfdataset import HFDataset
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.path_utils import get_output_path
from utils.model_utils import create_base_model, load_sequence_classification_model_from_dir, get_pooled_output
from config import EMBEDDING_SPACE_EMBEDDINGS_DIR, MODELS_SOURCES_DIR, GPU_DEVICE
from embedding_dataset import EmbeddingDataset


def compute_embeddings(trained_base_model, dataset, output_path, batch_size=64, overwrite=False):

    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    untrained_base_model = create_base_model()

    trained_base_model.to(device)
    untrained_base_model.to(device)

    trained_base_model.eval()
    untrained_base_model.eval()
    print('Loading models complete!')

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)

    os.makedirs(output_path, exist_ok=True)
    standard_embeddings_filepath = os.path.join(output_path, f'standard_embeddings.csv')
    trained_embeddings_filepath = os.path.join(output_path, f'trained_embeddings.csv')
    if os.path.isfile(standard_embeddings_filepath) and os.path.isfile(trained_embeddings_filepath) and not overwrite:
        print("Found embeddings.")
        return

    for embedding_filepath in [standard_embeddings_filepath, trained_embeddings_filepath]:
        if os.path.exists(embedding_filepath):
            os.remove(embedding_filepath)

    for step, batch in enumerate(tqdm(dataloader)):

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            trained_embeddings = get_pooled_output(trained_base_model, b_input_ids, b_input_mask).cpu().numpy()
            standard_embeddings = get_pooled_output(untrained_base_model, b_input_ids, b_input_mask).cpu().numpy()


        with open(standard_embeddings_filepath, "ab") as f:
            np.savetxt(f, standard_embeddings)

        with open(trained_embeddings_filepath, "ab") as f:
            np.savetxt(f, trained_embeddings)

    return EmbeddingDataset(output_path)


def compute_embeddings_source_model(dataset_name, num_source_samples='full',
                                    batch_size=64, overwrite=False, stream_datasets=False):
    model_path = get_output_path(MODELS_SOURCES_DIR,
                                 num_train_samples=num_source_samples,
                                 source_name=dataset_name)

    output_path = get_output_path(EMBEDDING_SPACE_EMBEDDINGS_DIR,
                                  num_train_samples=num_source_samples,
                                  source_name=dataset_name)

    standard_embeddings_filepath = os.path.join(output_path, f'standard_embeddings.csv')
    trained_embeddings_filepath = os.path.join(output_path, f'trained_embeddings.csv')
    if os.path.isfile(standard_embeddings_filepath) and os.path.isfile(trained_embeddings_filepath) and not overwrite:
        print("Found embeddings.")
        return

    # print(model_path)
    trained_model = load_sequence_classification_model_from_dir(model_path)
    # TODO: Write function in model_utils
    trained_base_model = getattr(trained_model, trained_model.base_model_prefix)

    dataset = HFDataset(dataset_name, split='train', max_num_examples=num_source_samples, streaming=stream_datasets)

    compute_embeddings(trained_base_model, dataset, output_path, batch_size=batch_size, overwrite=overwrite)
