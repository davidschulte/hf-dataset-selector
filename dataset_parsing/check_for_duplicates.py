from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from hfdataset import HFDataset
from . import dataset_info_dict
import pickle

target_datasets = [
    'imdb_plain_text',
    'tweet_eval_emotion',
    'tweet_eval_sentiment',
    'paws-x_en',
    'md_gender_bias_convai2_inferred',
    'llm-book__JGLUE_JSTS',
    'google_wellformed_query_default'
]

source_datasets = dataset_info_dict.keys()


def get_inputs_set(dataset_name, split='train', max_num_samples=10000):
    dataset = HFDataset(dataset_name, split=split, max_num_examples=max_num_samples)

    vocabulary_set = set()
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=256,
                            collate_fn=dataset.preprocess_inputs)

    for step, b_inputs in enumerate(dataloader):
        vocabulary_set = vocabulary_set.union(set(b_inputs))

    return vocabulary_set


def main():
    results_dict = {}
    for target_dataset in target_datasets:
        results_dict[target_dataset] = []
    target_inputs_sets = {target_dataset: get_inputs_set(target_dataset) for target_dataset in target_datasets}

    for source_dataset in tqdm(source_datasets):
        source_inputs_set = get_inputs_set(source_dataset)
        for target_dataset, target_inputs_set in target_inputs_sets.items():
            if target_inputs_set == source_inputs_set:
                print(f"{target_dataset} = {source_dataset}")
                results_dict[target_dataset].append(source_dataset)


    with open("duplicates.pkl", "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == '__main__':
    main()
