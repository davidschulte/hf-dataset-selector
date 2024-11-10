# hf-dataset-selecotr
[![PyPI version](https://img.shields.io/pypi/v/hf-dataset-selector.svg)](https://pypi.org/project/hf-dataset-selector)

A convenient and fast Python package to find the best datasets for intermediate fine-tuning for your task.

## How it works
hf-dataset-selector enables you to find good datasets from the Hugging Face Hub for intermediate fine-tuning before training on your task. It downloads small (~2.4MB each) neural networks for each intermediate task from the Hugging Face Hub. These neural networks are called Embedding Space Maps (ESMs) and transform embeddings produced by the language model. The transformed embeddings are ranked using LogME.

hf-dataset-selector only ranks datasets with a corresponding ESM on the Hugging Face Hub. We encourage you to train and publish your own ESMs for your datasets to enable others to rank them.


### What are Embedding Space Maps?
<!-- This section describes the evaluation protocols and provides the results. -->
Embedding Space Maps (ESMs) are neural networks that approximate the effect of fine-tuning a language model on a task. They can be used to quickly transform embeddings from a base model to approximate how a fine-tuned model would embed the the input text.
ESMs can be used for intermediate task selection with the ESM-LogME workflow.

## How to install

**hf-dataset-selector** is available on PyPi:

```bash
$ pip install hf-dataset-selector
```


## How to find suitable datasets for your problem

### Example

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

## How to train your own ESM
[TBD]

## How to cite


<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
If you are using this hf-dataset-selector, please cite our [paper](https://arxiv.org/abs/2410.15148).

**BibTeX:**


```
@misc{schulte2024moreparameterefficientselectionintermediate,
      title={Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning}, 
      author={David Schulte and Felix Hamborg and Alan Akbik},
      year={2024},
      eprint={2410.15148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.15148}, 
}
```


**APA:**

```
Schulte, D., Hamborg, F., & Akbik, A. (2024). Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning. arXiv preprint arXiv:2410.15148.
```

## How to reproduce the results from the paper
For reproducing the results of our paper, please refer to the [`emnlp-submission` branch](https://github.com/davidschulte/hf-dataset-selector/tree/emnlp-submission).


## Acknowledgements

Our methods extends the LogME method for intermediate task selection. We adapt the implementation by the authors.
https://github.com/tuvuumass/task-transferability
