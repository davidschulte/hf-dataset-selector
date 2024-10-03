from transfer_learning import execute_transfers
from dataset_parsing import dataset_info_dict
from tqdm import tqdm
import traceback
from config import TARGET_TASKS

freeze_base_model = False
source_dataset_names = dataset_info_dict.keys()
num_source_samples = 10000
num_target_samples = 1000
num_val_samples = 1000

all_dataset_exceptions = {}

for dataset_name in tqdm(source_dataset_names):

    try:
        execute_transfers(dataset_name,
                          target_dataset_names=TARGET_TASKS,
                          num_target_train_samples=num_target_samples,
                          num_source_samples=num_source_samples,
                          num_val_samples=num_val_samples,
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
