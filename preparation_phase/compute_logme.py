from torch.utils.data import SequentialSampler, DataLoader
from prediction_phase.LogME import LogME
import numpy as np
import torch
from utils.model_utils import get_pooled_output, EMBEDDING_SIZE
import os
from hfdataset import HFDataset
from tqdm import tqdm
import time
from datetime import datetime
import json
from utils.path_utils import get_output_path
from utils.model_utils import create_base_model, load_sequence_classification_model_from_dir
from utils.load_model_utils import load_model
from config import GPU_DEVICE, LOGME_TNN_DIR, LOGME_DIR, MODELS_SOURCES_DIR, TRANSFORMATION_NETS_FILEPATH, ESM_OPTIONAL_LAYER_DIMS


def compute_logme(dataset, base_model, regression, batch_size=128, transformation_net=None, scale_features=None):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)
    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
    base_model.to(device)

    if regression:
        label_dtype = float
    else:
        label_dtype = int

    labels = np.zeros(0, label_dtype)
    features = np.zeros((0, EMBEDDING_SIZE), float)

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.detach().cpu().numpy().flatten()

        with torch.no_grad():
            batch_embeddings = get_pooled_output(base_model, b_input_ids, b_input_mask)
            # batch_embeddings = base_model(b_input_ids, attention_mask=b_input_mask)[1]
            if transformation_net is not None:
                batch_embeddings = transformation_net(batch_embeddings)

        batch_embeddings = batch_embeddings.cpu().numpy()
        labels = np.append(labels, b_labels, axis=0)
        features = np.append(features, batch_embeddings, axis=0)

    logme = LogME(regression=regression, scale_features=scale_features)

    return logme.fit(features, labels)


def compute_logme_tnn_batch(dataset, base_model, transformation_nets, regression, batch_size=128,
                            feature_dim=EMBEDDING_SIZE, scale_features=None):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)
    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
    base_model.to(device)

    if regression:
        label_dtype = float
    else:
        label_dtype = int

    labels = np.zeros(0, label_dtype)
    features_list = [np.zeros((0, feature_dim), float)] * len(transformation_nets)

    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.detach().cpu().numpy().flatten()

        with torch.no_grad():
            batch_base_embeddings = get_pooled_output(base_model, b_input_ids, b_input_mask)
            for i, transformation_net in enumerate(transformation_nets):
                batch_transformed_embeddings = transformation_net(batch_base_embeddings).cpu().numpy()
                features_list[i] = np.append(features_list[i], batch_transformed_embeddings, axis=0)

        labels = np.append(labels, b_labels, axis=0)

    logme_results = []
    for features in tqdm(features_list):
        logme_results.append(LogME(regression=regression, scale_features=scale_features).fit(features, labels, add_intercept=False))

    return logme_results


def compute_logme_from_list(target_dataset_name, source_dataset_names, num_target_samples, num_source_samples,
                            seed=None, scale_features=None, timer_suffix=None):
        # results_dict = {}


    start_time = time.time()

    target_results = {}
    # max_num_examples = None if setting is None else 1000
    # seed = 42 if setting is None else setting
    target_dataset = HFDataset(target_dataset_name, split='train', max_num_examples=num_target_samples, seed=seed)
    regression = target_dataset.task_type == 'regression'

    for source_dataset_name in tqdm(source_dataset_names):
        source_model_dir = get_output_path(MODELS_SOURCES_DIR,
                                           # num_epochs=source_num_epochs,
                                           num_train_samples=num_source_samples,
                                           source_name=source_dataset_name)
        source_model = load_sequence_classification_model_from_dir(source_model_dir)
        source_base_model = getattr(source_model, source_model.base_model_prefix)
        # print(target_dataset_name, source_dataset_name)
        # source_bert = BertForSequenceClassification.from_pretrained(source_model_dir).bert
        logme_score = compute_logme(target_dataset, source_base_model, regression=regression,
                                    scale_features=scale_features)
        target_results[source_dataset_name] = logme_score

    time_elapsed = time.time() - start_time

    target_output_path = get_output_path(LOGME_DIR,
                                         num_train_samples=num_target_samples,
                                         num_source_samples=num_source_samples,
                                         seed=seed,
                                         target_name=target_dataset_name)

    for source_dataset_name in source_dataset_names:
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
        'num_sources': len(source_dataset_names),
        'device': used_device
    }

    if timer_suffix:
        timer_filename = f'timer_{timer_suffix}.json'
    else:
        timer_filename = 'timer.json'

    with open(os.path.join(target_output_path, timer_filename), 'w') as f:
        json.dump(timer_dict, f)

    # results_dict[target_dataset_name] = target_results

        # output_base_path = get_output_path(LOGME_DIR,
        #                                    num_train_samples=num_target_samples,
        #                                    num_source_samples=num_source_samples,
        #                                    seed=seed)
        # os.makedirs(output_base_path, exist_ok=True)

        # if return_dataframe:
        #     results_dict['Sources'] = source_dataset_names
        #
        #     df = pd.DataFrame().from_dict(results_dict).set_index('Sources')
        #     df.to_pickle(os.path.join(output_base_path, 'logme_df.pkl'))
        #


def compute_esm_logme_from_list(target_dataset_name, source_dataset_names, num_target_samples, num_source_samples,
                                use_tnn_bottleneck=False,
                                seed=None,
                                scale_features=None,
                                num_bins=None,
                                timer_suffix=None):

    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")
    base_model = create_base_model()
    base_model.to(device)

    start_time = time.time()
    # target_results = {}
    target_dataset = HFDataset(target_dataset_name, split='train', max_num_examples=num_target_samples, seed=seed)
    regression = target_dataset.task_type == 'regression'

    source_transformation_nets_paths = [os.path.join(get_output_path(TRANSFORMATION_NETS_FILEPATH,
                                                                     num_train_samples=num_source_samples,
                                                                     optional_layers=ESM_OPTIONAL_LAYER_DIMS),
                                                     f'{dataset_name}.pt') for dataset_name in source_dataset_names]
    source_transformation_nets = [load_model(filepath,
                                             model_type='transformation_network',
                                             device=device) for filepath in source_transformation_nets_paths]
    if use_tnn_bottleneck:
        feature_dim = source_transformation_nets[0].bottleneck_dim
        source_transformation_nets = [transformation_net.bottleneck_model for transformation_net in
                                      source_transformation_nets]
    else:
        feature_dim = EMBEDDING_SIZE

    target_results = compute_logme_tnn_batch(target_dataset,
                                             base_model,
                                             source_transformation_nets,
                                             # regression=False,
                                             regression=regression,
                                             feature_dim=feature_dim,
                                             scale_features=scale_features,
                                             num_bins=num_bins)

    time_elapsed = time.time() - start_time
# df = pd.DataFrame().from_dict(results_dict, orient='index', columns=results_dict.keys())
    target_output_path = get_output_path(LOGME_TNN_DIR,
                                         num_train_samples=num_target_samples,
                                         num_source_samples=num_source_samples,
                                         seed=seed,
                                         target_name=target_dataset_name,
                                         optional_layers=ESM_OPTIONAL_LAYER_DIMS)

    for source_i, source_dataset_name in enumerate(source_dataset_names):
        output_dir = get_output_path(target_output_path,
                                     source_name=source_dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'metric.npy'), target_results[source_i])

    device = torch.device(GPU_DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    if device.type == "cuda":
        used_device = torch.cuda.get_device_name(device)
    else:
        used_device = "cpu"

    timer_dict = {
        'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'elapsed': time_elapsed,
        'num_sources': len(source_dataset_names),
        'device': used_device
    }

    if timer_suffix:
        timer_filename = f'timer_{timer_suffix}.json'
    else:
        timer_filename = 'timer.json'

    with open(os.path.join(target_output_path, timer_filename), 'w') as f:
        json.dump(timer_dict, f)
