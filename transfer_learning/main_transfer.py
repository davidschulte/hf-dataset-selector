from transfer_learning import execute_transfers
from dataset_parsing import dataset_info_dict
from tqdm import tqdm
import traceback
from config import TARGET_TASKS, NUM_SOURCE_SAMPLES, NUM_TARGET_SAMPLES, NUM_VAL_SAMPLES

freeze_base_model = False
source_dataset_names = dataset_info_dict.keys()

all_dataset_exceptions = {}

for dataset_name in tqdm(source_dataset_names):

    try:
        execute_transfers(dataset_name,
                          target_dataset_names=TARGET_TASKS,
                          num_target_train_samples=NUM_TARGET_SAMPLES,
                          num_source_samples=NUM_SOURCE_SAMPLES,
                          num_val_samples=NUM_VAL_SAMPLES,
                          save_target_models=False,
                          overwrite_target_training=False,
                          overwrite_source_training=False,
                          overwrite_evaluation=False,
                          stream_source_dataset=True,
                          disable_tqdm=True,
                          freeze_base_model=freeze_base_model)

    except Exception as e:
        print(dataset_name)
        print(str(traceback.format_exc()))
