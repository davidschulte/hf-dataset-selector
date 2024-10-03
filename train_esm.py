import os
from torch.utils.data import SequentialSampler, DataLoader
import torch
from tqdm import tqdm
from embedding_dataset import EmbeddingDataset
from trainers import ESMTrainer
from utils.path_utils import get_output_path
from utils.load_model_utils import load_model
import time
from datetime import datetime
import json
from config import TRANSFORMATION_NETS_FILEPATH, EMBEDDINGS_BASE_DIR, ESM_OPTIONAL_LAYER_DIMS


def train_transformation_network(embeddings_dataset,
                                 output_filepath=None,
                                 num_epochs=10,
                                 optional_layer_dims=ESM_OPTIONAL_LAYER_DIMS,
                                 weight_decay=0.01,
                                 batch_size=32,
                                 overwrite=False):

    if output_filepath:
        if os.path.isfile(output_filepath) and not overwrite:
            print('Found transformation network on disk.')
            return load_model(output_filepath, 'transformation_network')

    sampler = SequentialSampler(embeddings_dataset)
    dataloader = DataLoader(embeddings_dataset, sampler=sampler, batch_size=batch_size)

    total_steps = len(dataloader) * num_epochs
    transformation_network_trainer = ESMTrainer(num_train_steps=total_steps,
                                                model_optional_layer_dims=optional_layer_dims,
                                                weight_decay=weight_decay)

    epoch_train_durations = []
    epoch_avg_losses = []
    start_time = time.time()
    for epoch_i in range(num_epochs):

        transformation_network_trainer.reset_loss()
        with tqdm(dataloader, desc=f'Training: Epoch {epoch_i} / {num_epochs}', unit='batch') as pbar:

            for step, batch in enumerate(pbar):

                loss = transformation_network_trainer.train_step(batch)

                avg_train_loss = loss / batch_size

                pbar.set_postfix(avg_train_loss=avg_train_loss)

        end_time = time.time()
        epoch_train_durations.append(end_time - start_time)
        start_time = end_time

        epoch_avg_losses.append(transformation_network_trainer.avg_loss)

    model = transformation_network_trainer.model

    if output_filepath:
        output_dir = os.path.dirname(output_filepath)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), output_filepath)
        train_info_dict = {
            'training_completed_timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            'num_epochs': num_epochs,
            'num_train_examples': len(embeddings_dataset),
            'epoch_train_durations': epoch_train_durations,
            'epoch_avg_losses': epoch_avg_losses
        }

        with open(os.path.join(output_dir, 'train_info.json'), 'w') as f:
            json.dump(train_info_dict, f)

        print('Saved model.')

    return model


def train_transformation_network_source_embeddings(dataset_name,
                                                   num_epochs=10,
                                                   optional_layer_dims=ESM_OPTIONAL_LAYER_DIMS,
                                                   weight_decay=0.001,
                                                   batch_size=32,
                                                   num_source_samples='full',
                                                   overwrite=False,
                                                   save_model=True,
                                                   return_model=True):
    if save_model:
        # output_filepath = os.path.join(TRANSFORMATION_NETS_FILEPATH, f'{dataset_name}.pt')
        output_filepath = os.path.join(get_output_path(TRANSFORMATION_NETS_FILEPATH,
                                                       num_train_samples=num_source_samples,
                                                       optional_layers=optional_layer_dims),
                                       # f'weight_decay_{weight_decay}'
                                       f'{dataset_name}.pt')

        if os.path.isfile(output_filepath) and not overwrite:
            print('Found transformation network on disk.')
            return load_model(output_filepath, 'transformation_network') if return_model else None

    else:
        output_filepath = None

    embeddings_filepath = get_output_path(EMBEDDINGS_BASE_DIR,
                                          num_train_samples=num_source_samples,
                                          source_name=dataset_name)
    # dataset = EmbeddingDataset(os.path.join(EMBEDDINGS_BASE_DIR, dataset_name))

    dataset = EmbeddingDataset(embeddings_filepath)

    transformation_network = train_transformation_network(dataset,
                                                          output_filepath=output_filepath,
                                                          num_epochs=num_epochs,
                                                          optional_layer_dims=optional_layer_dims,
                                                          weight_decay=weight_decay,
                                                          batch_size=batch_size,
                                                          overwrite=overwrite)

    return transformation_network if return_model else None
