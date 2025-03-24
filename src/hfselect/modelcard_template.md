---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# ESM {{ model_id | default("Model ID", true) }}

<!-- Provide a quick summary of what the model is/does. -->

{{ model_summary | default("", true) }}

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

{{ model_description | default("", true) }}

- **Developed by:** {{ developers | default("[Unknown]", true)}}
- **Model type:** ESM
- **Base Model:** {{ base_model | default("[More Information Needed]", true)}}
- **Intermediate Task:** {{ task_id | default("[More Information Needed]", true)}}
- **ESM architecture:** {{ esm_architecture | default("[More Information Needed] (The default architecture is a single dense layer.)", true)}}
- **ESM embedding dimension:** {{ esm_embedding_dim | default("[More Information Needed]", true)}}
- **Language(s) (NLP):** {{ language | default("[More Information Needed]", true)}}
- **License:** Apache-2.0 license
- **ESM version:** {{ version | default("[More Information Needed]", true)}}

## Training Details

### Intermediate Task
- **Task ID:** {{ task_id | default("[More Information Needed]", true)}}
- **Subset [optional]:** {{ task_subset | default("", true)}}
- **Text Column:** {{ text_column | default("", true)}}
- **Label Column:** {{ label_column | default("", true)}}
- **Dataset Split:**  {{task_split | default("[More Information Needed]", true)}}
- **Sample size [optional]:** {{num_examples | default("", true)}}
- **Sample seed [optional]:** {{seed | default("", true)}}

### Training Procedure [optional]

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Language Model Training Hyperparameters [optional]
- **Epochs:** {{ lm_num_epochs | default("[More Information Needed]", true)}}
- **Batch size:** {{ lm_batch_size | default("[More Information Needed]", true)}}
- **Learning rate:** {{ lm_learning_rate | default("[More Information Needed]", true)}}
- **Weight Decay:** {{ lm_weight_decay | default("[More Information Needed]", true)}}
- **Optimizer**: {{ lm_optimizer | default("[More Information Needed]", true)}}

### ESM Training Hyperparameters [optional]
- **Epochs:** {{ esm_num_epochs | default("[More Information Needed]", true)}}
- **Batch size:** {{ esm_batch_size | default("[More Information Needed]", true)}}
- **Learning rate:** {{ esm_learning_rate | default("[More Information Needed]", true)}}
- **Weight Decay:** {{ esm_weight_decay | default("[More Information Needed]", true)}}
- **Optimizer**: {{ esm_optimizer | default("[More Information Needed]", true)}}


### Additional trainiung details [optional]
{{ training_details | default("", true)}}

## Model evaluation

### Evaluation of fine-tuned language model [optional]
{{ evaluation_lm | default("", true)}}

### Evaluation of ESM [optional]
MSE: {{ esm_mse | default("", true)}}

### Additional evaluation details [optional]
{{ evaluation_details | default("", true)}}

## What are Embedding Space Maps used for?
Embedding Space Maps are a part of ESM-LogME, a efficient method for finding intermediate datasets for transfer learning. There are two reasons to use ESM-LogME:

### You don't have enough training data for your problem
If you don't have a enough training data for your problem, just use ESM-LogME to find more.
You can supplement model training by including publicly available datasets in the training process. 

1. Fine-tune a language model on suitable intermediate dataset.
2. Fine-tune the resulting model on your target dataset.

This workflow is called intermediate task transfer learning and it can significantly improve the target performance.

But what is a suitable dataset for your problem? ESM-LogME enable you to quickly rank thousands of datasets on the Hugging Face Hub by how well they are exptected to transfer to your target task.

### You want to find similar datasets to your target dataset
Using ESM-LogME can be used like search engine on the Hugging Face Hub. You can find similar tasks to your target task without having to rely on heuristics. ESM-LogME estimates how language models fine-tuned on each intermediate task would benefinit your target task. This quantitative approach combines the effects of domain similarity and task similarity. 

## How can I use ESM-LogME / ESMs?
[![PyPI version](https://img.shields.io/pypi/v/hf-dataset-selector.svg)](https://pypi.org/project/hf-dataset-selector)

We release **hf-dataset-selector**, a Python package for intermediate task selection using Embedding Space Maps.

**hf-dataset-selector** fetches ESMs for a given language model and uses it to find the best dataset for applying intermediate training to the target task. ESMs are found by their tags on the Huggingface Hub.

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

For more information on how to use ESMs please have a look at the [official Github repository](https://github.com/davidschulte/hf-dataset-selector). We provide documentation further documentation and tutorials for finding intermediate datasets and training your own ESMs.


## How do Embedding Space Maps work?

<!-- This section describes the evaluation protocols and provides the results. -->
Embedding Space Maps (ESMs) are neural networks that approximate the effect of fine-tuning a language model on a task. They can be used to quickly transform embeddings from a base model to approximate how a fine-tuned model would embed the the input text.
ESMs can be used for intermediate task selection with the ESM-LogME workflow.

## How can I use Embedding Space Maps for Intermediate Task Selection?

## Citation


<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
If you are using this Embedding Space Maps, please cite our [paper](https://aclanthology.org/2024.emnlp-main.529/).

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

## Additional Information

{{ additional_info | default("", true)}}