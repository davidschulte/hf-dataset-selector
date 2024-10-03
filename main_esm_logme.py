from compute_logme import compute_esm_logme_from_list
from dataset_parsing import dataset_info_dict
from config import TARGET_TASKS

source_dataset_names = dataset_info_dict.keys()
num_source_samples = 10000
num_target_samples = 1000
num_val_samples = 1000
transfer_exceptions = {}
for target_dataset_name in TARGET_TASKS:
    compute_esm_logme_from_list(target_dataset_name=target_dataset_name,
                                source_dataset_names=source_dataset_names,
                                num_target_samples=num_target_samples,
                                num_source_samples=num_source_samples)
