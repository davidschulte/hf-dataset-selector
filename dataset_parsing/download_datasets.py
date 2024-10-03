from . import dataset_info_dict
from hfdataset import HFDataset
from tqdm import tqdm

exceptions = {}

train_datasets = sorted(list(dataset_info_dict.keys()))
target_datasets = [
    'imdb_plain_text',
    'tweet_eval_emotion',
    'tweet_eval_sentiment',
    'llm-book__JGLUE_JSTS',
    'google_wellformed_query_default',
    'paws-x_en',
    'md_gender_bias_convai2_inferred',
    'google__civil_comments_default'
]
source_sample_list = [1000, 10000, 'full']
target_samples = 1000
validation_samples = 1000
target_seed_range = [None]

for ds_name in tqdm(train_datasets):

    for source_samples in source_sample_list:
        try:
            ds = HFDataset(ds_name, split="train", max_num_examples=source_samples, streaming=False)
            ds.save_locally()

        except Exception as e:
            print(str(e))
            exceptions[ds_name] = str(e)

for ds_name in tqdm(target_datasets):
    try:
        for seed in target_seed_range:
            ds = HFDataset(ds_name, split="train", seed=seed, max_num_examples=target_samples, streaming=False)
            ds.save_locally()
        ds = HFDataset(ds_name, split="validation", max_num_examples=validation_samples, streaming=False)
        ds.save_locally()
        # ds = HFDataset(ds_name)
#
    except Exception as e:
        print(str(e))
        exceptions[ds_name] = str(e)
