import os.path
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedModel
from hfdataset import HFDataset
from tqdm import tqdm
import time
from datetime import datetime
import json
import pickle
from utils.path_utils import get_output_path
from utils.model_utils import reset_head_layer, load_sequence_classification_model_from_dir
from trainers import TransformerTrainer
from evaluators import TransformerEvaluator
from config import MODELS_TRANSFER_DIR, MODELS_SOURCES_DIR, EVAL_TRANSFER_DIR, MODELS_FROZEN_TRANSFER_DIR, EVAL_FROZEN_TRANSFER_DIR


def train_model(dataset, label_dim, model=None, batch_size=32, num_epochs=3, train_info_output_dir=None,
                freeze_base_model=False, disable_tqdm=True):
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  collate_fn=dataset.collate_fn)

    num_train_steps = len(train_dataloader) * num_epochs

    trainer = TransformerTrainer(output_dim=label_dim,
                                 num_train_steps=num_train_steps,
                                 model=model,
                                 freeze_base_model=freeze_base_model)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    epoch_train_durations = []
    epoch_avg_losses = []
    start_time = time.time()

    for epoch_i in range(1, num_epochs+1):
        trainer.reset_loss()
        with tqdm(train_dataloader, desc=f'Training: Epoch {epoch_i} / {num_epochs}', unit='batch',
                  disable=disable_tqdm) as pbar:
            for step, batch in enumerate(pbar, start=1):
                loss = trainer.train_step(batch)

                avg_train_loss = loss / batch_size

                pbar.set_postfix(avg_train_loss=avg_train_loss)

        end_time = time.time()
        epoch_train_durations.append(end_time - start_time)
        start_time = end_time

        epoch_avg_losses.append(trainer.avg_loss)


    if train_info_output_dir is not None:
        os.makedirs(train_info_output_dir, exist_ok=True)

        if trainer.device.type == "cuda":
            used_device = torch.cuda.get_device_name(trainer.device)
        else:
            used_device = "cpu"

        train_info_dict = {
            'training_completed_timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            'num_epochs': num_epochs,
            'num_train_examples': len(dataset),
            'epoch_train_durations': epoch_train_durations,
            'epoch_avg_losses': epoch_avg_losses,
            'device': used_device
        }

        with open(os.path.join(train_info_output_dir, 'train_info.json'), 'w') as f:
            json.dump(train_info_dict, f)

    return trainer.model


def validate_model(model, dataset, batch_size=2048, disable_tqdm=True):
    validation_sampler = SequentialSampler(dataset)
    validation_dataloader = DataLoader(dataset,
                                       sampler=validation_sampler,
                                       batch_size=batch_size,
                                       collate_fn=dataset.collate_fn)

    evaluator = TransformerEvaluator(model=model, metric=dataset.metric)

    with tqdm(validation_dataloader, desc='Validating Model', unit='batch', disable=disable_tqdm) as pbar:
        for batch in pbar:
            # batch_accuracy = evaluator.evaluate_step(batch)
            # pbar.set_postfix(batch_accuracy=batch_accuracy)
            evaluator.evaluate_step(batch)

    results = evaluator.evaluation_results

    # print(f'Accuracy: {results["metric"]:0.4f}')

    return results

def transfer(model: PreTrainedModel, dataset, num_classes, num_epochs=3, train_info_output_dir=None, freeze_base_model=False,
             disable_tqdm=True):
    # model = deepcopy(model)
    model = reset_head_layer(model, num_classes)
    model = train_model(dataset, num_classes, model=model, num_epochs=num_epochs, train_info_output_dir=train_info_output_dir,
                        freeze_base_model=freeze_base_model, disable_tqdm=disable_tqdm)

    return model


def execute_transfers(source_dataset_name: str, target_dataset_names: list, num_epochs=3,
                      num_target_train_samples='full', num_source_samples='full', num_val_samples='full',
                      save_target_models=False,
                      overwrite_source_training=False, overwrite_target_training=False, overwrite_evaluation=False, seed=None,
                      stream_source_dataset=False, stream_target_datasets=False, freeze_base_model=False,
                      disable_tqdm=True):
    source_model_dir = get_output_path(MODELS_SOURCES_DIR,
                                       # num_epochs=num_epochs,
                                       num_train_samples=num_source_samples,
                                       source_name=source_dataset_name)

    eval_dir = EVAL_FROZEN_TRANSFER_DIR if freeze_base_model else EVAL_TRANSFER_DIR

    # If there is already an evaluation, do not overwrite it
    if not overwrite_source_training and not overwrite_target_training and not overwrite_evaluation:
        eval_filepaths = [get_output_path(eval_dir,
                                          num_train_samples=num_target_train_samples,
                                          num_source_samples=num_source_samples,
                                          seed=seed,
                                          source_name=None,
                                          target_name=target_dataset_name,
                                          filename=f'{source_dataset_name}.pkl')
                          for target_dataset_name in target_dataset_names]

        if all([os.path.isfile(eval_filepath) for eval_filepath in eval_filepaths]):
            return

    if os.path.isdir(source_model_dir) and not overwrite_source_training:
        # source_model = BertForSequenceClassification.from_pretrained(source_model_dir)
        # print(f'Loading source model: {source_dataset_name}')
        source_model = load_sequence_classification_model_from_dir(source_model_dir)
    else:

        source_dataset = HFDataset(source_dataset_name, split='train', max_num_examples=num_source_samples,
                                   streaming=stream_source_dataset)
        source_label_dim = source_dataset.label_dim
        source_model = train_model(source_dataset, source_label_dim, num_epochs=num_epochs,
                                   train_info_output_dir=source_model_dir, disable_tqdm=disable_tqdm)
        source_model.save_pretrained(source_model_dir)

    for target_dataset_name in target_dataset_names:
        if target_dataset_name == source_dataset_name:
            get_single_model_performance(source_dataset_name,
                                         num_epochs=num_epochs,
                                         num_train_samples=num_target_train_samples,
                                         num_source_samples=num_source_samples,
                                         num_val_samples=num_val_samples,
                                         overwrite_training=overwrite_target_training,
                                         overwrite_evaluation=overwrite_evaluation,
                                         save_model=save_target_models,
                                         seed=seed,
                                         freeze_base_model=freeze_base_model,
                                         disable_tqdm=disable_tqdm)
            continue

        results_dir = get_output_path(eval_dir,
                                      # num_epochs=num_epochs,
                                      num_train_samples=num_target_train_samples,
                                      num_source_samples=num_source_samples,
                                      seed=seed,
                                      source_name=None,
                                      target_name=target_dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        results_filepath = os.path.join(results_dir, f'{source_dataset_name}.pkl')

        # If there is already an evaluation, do not overwrite it
        if os.path.isfile(results_filepath) and not overwrite_evaluation:
            continue

        target_train_dataset = HFDataset(target_dataset_name,
                                         split='train',
                                         max_num_examples=num_target_train_samples,
                                         seed=seed,
                                         streaming=stream_target_datasets)
        target_label_dim = target_train_dataset.label_dim

        if freeze_base_model:
            model_dir = MODELS_FROZEN_TRANSFER_DIR
            target_train_dataset, target_eval_dataset = target_train_dataset.train_test_split()
        else:
            model_dir = MODELS_TRANSFER_DIR
            target_eval_dataset = HFDataset(target_dataset_name, split='validation', streaming=stream_target_datasets,
                                            max_num_examples=num_val_samples)

        transferred_model_dir = get_output_path(model_dir,
                                                # num_epochs=num_epochs,
                                                num_train_samples=num_target_train_samples,
                                                num_source_samples=num_source_samples,
                                                seed=seed,
                                                source_name=source_dataset_name,
                                                target_name=target_dataset_name)

        if os.path.isdir(transferred_model_dir) and not overwrite_target_training:
            transferred_model = load_sequence_classification_model_from_dir(transferred_model_dir)
        else:
            train_info_output_dir = transferred_model_dir if save_target_models else None
            transferred_model = transfer(source_model,
                                         target_train_dataset,
                                         target_label_dim,
                                         num_epochs=num_epochs,
                                         train_info_output_dir=train_info_output_dir,
                                         freeze_base_model=freeze_base_model,
                                         disable_tqdm=disable_tqdm)
            if save_target_models:
                transferred_model.save_pretrained(transferred_model_dir)

        evaluation_results = validate_model(transferred_model, target_eval_dataset, disable_tqdm=disable_tqdm)

        with open(results_filepath, 'wb') as f:
            pickle.dump(evaluation_results, f)


def get_single_model_performance(dataset_name: str, num_epochs=3,
                                 num_train_samples='full', num_source_samples='full',
                                 num_val_samples='full',
                                 overwrite_training=False, overwrite_evaluation=False,
                                 save_model=True,
                                 seed=None,
                                 stream_datasets=False,
                                 freeze_base_model=False,
                                 disable_tqdm=True):
    print(f'Training and evaluating model on dataset: {dataset_name}')
    train_dataset = HFDataset(dataset_name, split='train', max_num_examples=num_train_samples, seed=seed,
                              streaming=stream_datasets)
    label_dim = train_dataset.label_dim
    test_dataset = HFDataset(dataset_name, split='validation', max_num_examples=num_val_samples)

    model_base_dir = MODELS_FROZEN_TRANSFER_DIR if freeze_base_model else MODELS_SOURCES_DIR
    model_dir = get_output_path(model_base_dir,
                                num_train_samples=num_train_samples,
                                source_name=dataset_name)

    if os.path.isdir(model_dir) and not overwrite_training:
        model = load_sequence_classification_model_from_dir(model_dir)
    else:
        train_info_output_dir = model_dir if save_model else None
        model = train_model(train_dataset, label_dim, num_epochs=num_epochs,
                            train_info_output_dir=train_info_output_dir,
                            freeze_base_model=freeze_base_model,
                            disable_tqdm=disable_tqdm)
        if save_model:
            model.save_pretrained(model_dir)

    results_base_dir = EVAL_FROZEN_TRANSFER_DIR if freeze_base_model else EVAL_TRANSFER_DIR
    results_dir = get_output_path(results_base_dir,
                                  num_train_samples=num_train_samples,
                                  num_source_samples=num_source_samples,
                                  seed=seed,
                                  source_name=None,
                                  target_name=dataset_name)

    os.makedirs(results_dir, exist_ok=True)

    results_filepath = os.path.join(results_dir, f'{dataset_name}.pkl')
    if os.path.isfile(results_filepath) and not overwrite_evaluation:
        return

    evaluation_results = validate_model(model, test_dataset, disable_tqdm=disable_tqdm)
    with open(results_filepath, 'wb') as f:
        pickle.dump(evaluation_results, f)


def get_single_model_performances(dataset_names: list, num_epochs=3, overwrite_training=False,
                                  disable_tqdm=True):
    print('Evaluating list datasets.')
    for dataset_name in dataset_names:
        get_single_model_performance(dataset_name,
                                     num_epochs=num_epochs,
                                     overwrite_training=overwrite_training,
                                     disable_tqdm=disable_tqdm)

    print('Evaluation done. Saved results.')


def train_model_from_dataset(dataset_name, num_train_samples='full', num_epochs=3, overwrite_training=False, seed=None,
                             stream_datasets=False, return_model=True, disable_tqdm=True):
    model_dir = get_output_path(MODELS_SOURCES_DIR,
                                # num_epochs=num_epochs,
                                num_train_samples=num_train_samples,
                                seed=seed,
                                source_name=dataset_name)

    if os.path.isdir(model_dir) and not overwrite_training:
        return load_sequence_classification_model_from_dir(model_dir) if return_model else None

    print(f'Training model on dataset: {dataset_name}')

    train_dataset = HFDataset(dataset_name, split='train', max_num_examples=num_train_samples, seed=seed,
                              streaming=stream_datasets)
    label_dim = train_dataset.label_dim

    model = train_model(train_dataset, label_dim, num_epochs=num_epochs, train_info_output_dir=model_dir,
                        disable_tqdm=disable_tqdm)
    model.save_pretrained(model_dir)

    return model if return_model else None
