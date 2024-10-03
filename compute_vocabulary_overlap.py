from hfdataset import HFDataset
from torch.utils.data import DataLoader, SequentialSampler


def compute_vocabulary_overlap(dataset1: HFDataset, dataset2: HFDataset, batch_size=256):
    vocabulary1 = get_vocabulary_set(dataset1)
    vocabulary2 = get_vocabulary_set(dataset2)

    return compute_jaccard_index(vocabulary1, vocabulary2)


def get_vocabulary_set(dataset, batch_size=512):
    vocabulary_set = set()
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=batch_size,
                            collate_fn=dataset.collate_fn)

    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0]
        vocabulary_set = vocabulary_set.union(set(b_input_ids.flatten().tolist()))

    return vocabulary_set


def compute_jaccard_index(set1, set2):
    overlap = set1.intersection(set2)
    union = set1.union(set2)

    return len(overlap) / len(union)
