# hf-dataset-selector
[![PyPI version](https://img.shields.io/pypi/v/hf-dataset-selector.svg)](https://pypi.org/project/hf-dataset-selector)

A convenient and fast Python package to find the best datasets for intermediate fine-tuning for your task.

## Why hf-dataset-selector?
### You don't have enough training data for your problem
If you don't have a enough training data for your problem, just use hf-dataset-selector to find more.
You can supplement model training by including publicly available datasets in the training process. 

1. Fine-tune a language model on suitable intermediate dataset.
2. Fine-tune the resulting model on your target dataset.

This workflow is called intermediate task transfer learning and it can significantly improve the target performance.

But what is a suitable dataset for your problem? hf-dataset-selector enables you to quickly rank thousands of datasets on the Hugging Face Hub by how well they are exptected to transfer to your target task. Just specify a base language model and your target dataset, and hf-dataset-selector produces a ranking of intermediate datasets.

### You want to find similar datasets to your target dataset
hf-dataset-selector can be used like search engine on the Hugging Face Hub. You can find similar tasks to your target task without having to rely on heuristics. hf-dataset-selector estimates how language models fine-tuned on each intermediate task would benefinit your target task. This quantitative approach combines the effects of domain similarity and task similarity. 

## How to install

**hf-dataset-selector** is available on PyPi:

```bash
$ pip install hf-dataset-selector
```


## Quickstart

### How to find suitable datasets for your problem

```python
from hfselect import Dataset, compute_task_ranking

# Load target dataset from the Hugging Face Hub
dataset = Dataset.from_hugging_face(
    name="stanfordnlp/imdb",
    split="train",
    text_col="text",
    label_col="label",
    is_regression=False,
    num_examples=1000,
    seed=42
)

# Fetch ESMs and rank tasks
task_ranking = compute_task_ranking(
    dataset=dataset,
    model_name="bert-base-multilingual-uncased"
)

# Display top 5 recommendations
print(task_ranking[:5])
```
```python
1.   davanstrien/test_imdb_embedd2                     Score: -0.618529
2.   davanstrien/test_imdb_embedd                      Score: -0.618644
3.   davanstrien/test1                                 Score: -0.619334
4.   stanfordnlp/imdb                                  Score: -0.619454
5.   stanfordnlp/sst                                   Score: -0.62995
```

|   Rank | Task ID                       | Task Subset     | Text Column   | Label Column   | Task Split   |   Num Examples | ESM Architecture   |     Score |
|-------:|:------------------------------|:----------------|:--------------|:---------------|:-------------|---------------:|:-------------------|----------:|
|      1 | davanstrien/test_imdb_embedd2 | default         | text          | label          | train        |          10000 | linear             | -0.618529 |
|      2 | davanstrien/test_imdb_embedd  | default         | text          | label          | train        |          10000 | linear             | -0.618644 |
|      3 | davanstrien/test1             | default         | text          | label          | train        |          10000 | linear             | -0.619334 |
|      4 | stanfordnlp/imdb              | plain_text      | text          | label          | train        |          10000 | linear             | -0.619454 |
|      5 | stanfordnlp/sst               | dictionary      | phrase        | label          | dictionary   |          10000 | linear             | -0.62995  |
|      6 | stanfordnlp/sst               | default         | sentence      | label          | train        |           8544 | linear             | -0.63312  |
|      7 | kuroneko5943/snap21           | CDs_and_Vinyl_5 | sentence      | label          | train        |           6974 | linear             | -0.634365 |
|      8 | kuroneko5943/snap21           | Video_Games_5   | sentence      | label          | train        |           6997 | linear             | -0.638787 |
|      9 | kuroneko5943/snap21           | Movies_and_TV_5 | sentence      | label          | train        |           6989 | linear             | -0.639068 |
|     10 | fancyzhx/amazon_polarity      | amazon_polarity | content       | label          | train        |          10000 | linear             | -0.639718 |
## Tutorials
We provide tutorials for finding intermediate datasets, and for training your own ESM for others to rank.

- [Tutorial 1: Rank  intermediate datasets](tutorials/01_find_datasets.ipynb)
- [Tutorial 2: Filter the pool of intermediate datasets / ESMs](tutorials/02_filter_esms.ipynb)
- [Tutorial 3: Train your own ESM](tutorials/03_train_esm.ipynb)
- [Tutorial 4: Advanced ESM training with hyper-parameter optimization](tutorials/04_advanced_esm_training.ipynb)

## Documentation
We host a [documentation on Read the Docs](https://hf-dataset-selector.readthedocs.io/en/latest/).

## How it works
hf-dataset-selector enables you to find good datasets from the Hugging Face Hub for intermediate fine-tuning before training on your task. It downloads small (~2.4MB each) neural networks for each intermediate task from the Hugging Face Hub. These neural networks are called Embedding Space Maps (ESMs) and transform embeddings produced by the language model. The transformed embeddings are ranked using LogME.

hf-dataset-selector ranks only datasets with a corresponding ESM on the Hugging Face Hub. We encourage you to train and publish your own ESMs for your datasets to enable others to rank them.


### What are Embedding Space Maps?

Embedding Space Maps (ESMs) are neural networks that approximate the effect of fine-tuning a language model on a task. They can be used to quickly transform embeddings from a base model to approximate how a fine-tuned model would embed the the input text.
ESMs can be used for intermediate task selection with the ESM-LogME workflow.

<div style="text-align: center;">
  <img src="esm_illustration.png" width="300" height="300" />
</div>


## How to cite


<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
If you are using this hf-dataset-selector, please cite our [paper](https://aclanthology.org/2024.emnlp-main.529/).

**BibTeX:**


```
@inproceedings{schulte-etal-2024-less,
    title = "Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning",
    author = "Schulte, David  and
      Hamborg, Felix  and
      Akbik, Alan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.529/",
    doi = "10.18653/v1/2024.emnlp-main.529",
    pages = "9431--9442",
    abstract = "Intermediate task transfer learning can greatly improve model performance. If, for example, one has little training data for emotion detection, first fine-tuning a language model on a sentiment classification dataset may improve performance strongly. But which task to choose for transfer learning? Prior methods producing useful task rankings are infeasible for large source pools, as they require forward passes through all source language models. We overcome this by introducing Embedding Space Maps (ESMs), light-weight neural networks that approximate the effect of fine-tuning a language model. We conduct the largest study on NLP task transferability and task selection with 12k source-target pairs. We find that applying ESMs on a prior method reduces execution time and disk space usage by factors of 10 and 278, respectively, while retaining high selection performance (avg. regret@5 score of 2.95)."
}
```


**APA:**

```
Schulte, D., Hamborg, F., & Akbik, A. (2024, November). Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 9431-9442).
```

## How to reproduce the results from the paper
For reproducing the results of our paper, please refer to the [`emnlp-submission` branch](https://github.com/davidschulte/hf-dataset-selector/tree/emnlp-submission).


## Acknowledgements

Our methods extends the LogME method for intermediate task selection. We adapt the implementation by the authors.
https://github.com/tuvuumass/task-transferability
