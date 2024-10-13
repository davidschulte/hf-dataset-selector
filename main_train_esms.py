from transfer_learning import train_model_from_dataset
from compute_embeddings import compute_embeddings_source_model
from train_esm import train_transformation_network_source_embeddings
from dataset_parsing import dataset_info_dict
from tqdm import tqdm
import traceback
from config import NUM_SOURCE_SAMPLES

dataset_names = list(dataset_info_dict.keys())

all_dataset_exceptions = {}
for _, dataset_name in enumerate(tqdm(dataset_names)):
    print(f'Training on dataset {dataset_name}')
    dataset_exceptions = {}
    try:
        train_model_from_dataset(dataset_name, num_train_samples=NUM_SOURCE_SAMPLES, stream_datasets=True,
                                 return_model=False)
    except Exception as e:
        dataset_exceptions['train_exception'] = traceback.format_exc()

    try:
        compute_embeddings_source_model(dataset_name, num_source_samples=NUM_SOURCE_SAMPLES)
    except Exception as e:
        pass

    try:
        train_transformation_network_source_embeddings(dataset_name,
                                                       num_source_samples=NUM_SOURCE_SAMPLES,
                                                       return_model=False)
    except Exception as e:
        pass
        dataset_exceptions['transformation_net_exception'] = traceback.format_exc()
        dataset_exceptions['transformation_net_exception'] = str(e)

    if len(dataset_exceptions) > 0:
        all_dataset_exceptions[dataset_name] = dataset_exceptions
        print('Exceptions occurred:')
        print(dataset_name)
        print(dataset_exceptions)