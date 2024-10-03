from utils.run_taskemb_CR import compute_taskemb, get_args
from transformers_taskemb import BertForSequenceClassification_TaskEmbeddings as BertForSequenceClassification

from hfdataset import HFDataset
import os
from utils.path_utils import get_output_path
import torch
from config import TASKEMB_EMBEDDINGS_DIR
import time
from datetime import datetime
import json
from tqdm import tqdm
from config import MODEL_NAME
from utils.model_utils import MODELS
import traceback

from dataset_parsing import dataset_info_dict

overwrite_embeddings = False
streaming = True

device_name = 'cuda:1'
bert_pretrained_name = MODELS[MODEL_NAME]['pretrained_name']

num_target_samples = 1000


target_dataset_names = ['imdb_plain_text', 'tweet_eval_emotion', 'tweet_eval_sentiment',
                        'llm-book__JGLUE_JSTS', 'google_wellformed_query_default', 'paws-x_en',
                        'md_gender_bias_convai2_inferred', 'google__civil_comments_default']
source_dataset_names = list(dataset_info_dict.keys())
num_source_samples_list = [10000]
seed_range = [None]

device = torch.device(device_name) if torch.cuda.is_available() else torch.device("cpu")

args = get_args()
args.n_gpu = torch.cuda.device_count()
args.model_type = 'bert'
args.finetune_classifier = True
args.finetune_feature_extractor = True
args.num_epochs = 3

output_base_dir = TASKEMB_EMBEDDINGS_DIR if args.finetune_feature_extractor else TASKEMB_EMBEDDINGS_DIR+'_frozen'


def main():
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    print(args)
    # print(device)
    print(torch.cuda.get_device_name(args.device))

    if args.scenario == 'sources':
        ds_list = source_dataset_names
        num_samples_list = num_source_samples_list
    else:
        ds_list = target_dataset_names
        num_samples_list = [num_target_samples]

    if args.dataset_start_idx:
        ds_list = ds_list[args.dataset_start_idx:]
    if args.dataset_end_idx:
        ds_list = ds_list[:args.dataset_end_idx]

    for num_samples in num_samples_list:
        iteration_seed_range = [None] if num_samples == 'full' else seed_range
        for seed in tqdm(iteration_seed_range):
            for dataset_name in tqdm(ds_list):
                args.output_dir = get_output_path(output_base_dir,
                                                  target_name=dataset_name,
                                                  num_train_samples=num_samples,
                                                  seed=seed)

                if os.path.isdir(args.output_dir) and not overwrite_embeddings:
                    continue

                print(f'Dataset {dataset_name}')
                try:
                    dataset = HFDataset(dataset_name, split='train', max_num_examples=num_samples, seed=seed,
                                        streaming=streaming)
                except:
                    try:
                        dataset = HFDataset(dataset_name, split='train', max_num_examples=num_samples, seed=seed,
                                            streaming=False)
                    except:
                        print(f'{dataset_name} failed!')
                        print(traceback.format_exc())
                        continue

                try:
                    model = BertForSequenceClassification.from_pretrained(
                        bert_pretrained_name,  # Use the 12-layer BERT model, with an uncased vocab.
                        num_labels=dataset.label_dim,  # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
                        output_attentions=False,  # Whether the model returns attentions weights.
                        output_hidden_states=False,  # Whether the model returns all hidden-states.
                    )

                    model.to(device)
                    os.makedirs(args.output_dir, exist_ok=True)
                    start_time = time.time()
                    task_emb = compute_taskemb(args=args, train_dataset=dataset, model=model)
                    time_elapsed = time.time() - start_time

                    if args.device.type == "cuda":
                        used_device = torch.cuda.get_device_name(args.device)
                    else:
                        used_device = "cpu"

                    timer_dict = {
                        'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                        'elapsed': time_elapsed,
                        'device': used_device
                    }

                    with open(os.path.join(args.output_dir, 'timer.json'), 'w') as f:
                        json.dump(timer_dict, f)
                except:
                    print(traceback.format_exc())
                    pass


if __name__ == "__main__":
    main()