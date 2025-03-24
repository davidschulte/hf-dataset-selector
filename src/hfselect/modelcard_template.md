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


## What are Embedding Space Maps?

<!-- This section describes the evaluation protocols and provides the results. -->
Embedding Space Maps (ESMs) are neural networks that approximate the effect of fine-tuning a language model on a task. They can be used to quickly transform embeddings from a base model to approximate how a fine-tuned model would embed the the input text.
ESMs can be used for intermediate task selection with the ESM-LogME workflow.

## How can I use Embedding Space Maps for Intermediate Task Selection?
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

For more information on how to use ESMs please have a look at the [official Github repository](https://github.com/davidschulte/hf-dataset-selector).

## Citation


<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
If you are using this Embedding Space Maps, please cite our [paper](https://arxiv.org/abs/2410.15148).

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

## Additional Information

{{ additional_info | default("", true)}}