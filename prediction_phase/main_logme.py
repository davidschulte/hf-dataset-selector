from preparation_phase.compute_logme import compute_logme_from_list
from dataset_parsing import dataset_info_dict
from config import TARGET_TASKS, NUM_SOURCE_SAMPLES, NUM_TARGET_SAMPLES

source_dataset_names = dataset_info_dict.keys()
transfer_exceptions = {}
for target_dataset_name in TARGET_TASKS:
    compute_logme_from_list(target_dataset_name=target_dataset_name,
                            source_dataset_names=source_dataset_names,
                            num_target_samples=NUM_TARGET_SAMPLES,
                            num_source_samples=NUM_SOURCE_SAMPLES)
