import pytest
from hfselect import Dataset
from transformers import AutoModel, AutoTokenizer

BERT_MODEL_NAME = "bert-base-multilingual-uncased"


@pytest.fixture(scope="session")
def imdb_dataset():
    """Load a sample of the IMDB dataset"""

    return Dataset.from_hugging_face(
        name="imdb",
        split="train",
        text_col="text",
        label_col="label",
        is_regression=False,
        num_examples=1000
    )


@pytest.fixture(scope="session")
def bert_model():
    return AutoModel.from_pretrained(BERT_MODEL_NAME)


@pytest.fixture(scope="session")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


@pytest.fixture(scope="session")
def bert_model_name():
    return BERT_MODEL_NAME
