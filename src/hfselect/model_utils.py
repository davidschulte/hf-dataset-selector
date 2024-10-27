import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, \
    BertModel, BertConfig,  BertForSequenceClassification, BertTokenizer,\
    RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,\
    DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, logging
from copy import deepcopy

SEQUENCE_CLASSIFICATION_MODEL_ARGS = {
    'output_attentions': False,
    'output_hidden_states': False
}

MODELS = {
    'bert': {
        'base_model': BertModel,
        'sequence_classification_model': BertForSequenceClassification,
        'tokenizer': BertTokenizer,
        'config': BertConfig,
        'pretrained_name': 'bert-base-uncased',
        'base_model_attribute_name': 'bert',
        'embedding_size': 768,
        'pooling_method': 'bert_pooling'
    },
    'bert_multilang': {
        'base_model': BertModel,
        'sequence_classification_model': BertForSequenceClassification,
        'tokenizer': BertTokenizer,
        'config': BertConfig,
        'pretrained_name': 'bert-base-multilingual-uncased',
        'base_model_attribute_name': 'bert',
        'embedding_size': 768,
        'pooling_method': 'bert_pooling'
    },
    'bert-tiny': {
        'base_model': BertModel,
        'sequence_classification_model': BertForSequenceClassification,
        'tokenizer': BertTokenizer,
        'config': BertConfig,
        'pretrained_name': 'prajjwal1/bert-tiny',
        'base_model_attribute_name': 'bert',
        'embedding_size': 128,
        'pooling_method': 'bert_pooling'
    },
    'roberta-base': {
        'base_model': RobertaModel,
        'sequence_classification_model': RobertaForSequenceClassification,
        'tokenizer': RobertaTokenizer,
        'config': RobertaConfig,
        'pretrained_name': 'roberta-base',
        'base_model_attribute_name': 'roberta',
        'embedding_size': 768,
        'pooling_method': 'roberta_pooling'
    },
    'distilbert': {
        'base_model': DistilBertModel,
        'sequence_classification_model': DistilBertForSequenceClassification,
        'tokenizer': DistilBertTokenizer,
        'config': DistilBertConfig,
        'pretrained_name': 'distilbert-base-uncased',
        'base_model_attribute_name': 'distilbert',
        'embedding_size': 768,
        'pooling_method': 'distilbert_pooling'
    },
    'sentence_mpnet': {
        'base_model': BertModel,
        'sequence_classification_model': BertForSequenceClassification,
        'tokenizer': AutoTokenizer,
        'config': BertConfig,
        'pretrained_name': 'sentence-transformers/all-mpnet-base-v2',
        'base_model_attribute_name': 'bert',
        'embedding_size': 768,
        'pooling_method': 'sbert_pooling'
    },
    'sentence_minilm': {
        'base_model': BertModel,
        'sequence_classification_model': BertForSequenceClassification,
        'tokenizer': AutoTokenizer,
        'config': BertConfig,
        'pretrained_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'base_model_attribute_name': 'bert',
        'embedding_size': 384,
        'pooling_method': 'sbert_pooling'
    },
    'sentence_minilm_multilang': {
        'base_model': BertModel,
        'sequence_classification_model': BertForSequenceClassification,
        'tokenizer': AutoTokenizer,
        'config': BertConfig,
        'pretrained_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'base_model_attribute_name': 'bert',
        'embedding_size': 384,
        'pooling_method': 'sbert_pooling'
    }
}


def create_base_model(model_name):
    logging.set_verbosity_error()
    base_model_class: PreTrainedModel = MODELS[model_name]['base_model']
    pretrained_name = MODELS[model_name]['pretrained_name']

    return base_model_class.from_pretrained(pretrained_name)


# def create_sequence_classification_model(num_labels, model_name=MODEL_NAME):
#     logging.set_verbosity_error()
#     sequence_classification_model_class: PreTrainedModel = MODELS[model_name]['sequence_classification_model']
#     pretrained_name = MODELS[model_name]['pretrained_name']
#
#     return sequence_classification_model_class.from_pretrained(pretrained_name,
#                                                                num_labels=num_labels,
#                                                                **SEQUENCE_CLASSIFICATION_MODEL_ARGS)


def create_tokenizer(model_name):
    tokenizer_class: PreTrainedTokenizer = MODELS[model_name]['tokenizer']
    pretrained_name = MODELS[model_name]['pretrained_name']

    return tokenizer_class.from_pretrained(pretrained_name)
#
#
# def load_sequence_classification_model_from_dir(model_dir, model_name=MODEL_NAME):
#     sequence_classification_model_class: PreTrainedModel = MODELS[model_name]['sequence_classification_model']
#
#     return sequence_classification_model_class.from_pretrained(model_dir)
#
#
# def reset_head_layer(model: PreTrainedModel, num_labels, model_name=MODEL_NAME):
#     logging.set_verbosity_error()
#     # model_name = model.base_model_prefix
#     config_class = MODELS[model_name]['config']
#     pretrained_name = MODELS[model_name]['pretrained_name']
#     sequence_classification_model_class = MODELS[model_name]['sequence_classification_model']
#     base_model_attribute_name = MODELS[model_name]['base_model_attribute_name']
#     new_model_config = config_class.from_pretrained(pretrained_name, num_labels=num_labels)
#     new_model = sequence_classification_model_class(new_model_config)
#     # new_model = sequence_classification_model_class(model.config)
#     setattr(new_model, base_model_attribute_name, deepcopy(getattr(model, base_model_attribute_name)))
#
#     return new_model


def get_pooled_output(
        base_model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor):
    # pooling_method = MODELS[MODEL_NAME]['pooling_method']

    if isinstance(base_model, BertModel):
        if "sentence-transformers" in base_model.name_or_path:
            token_embeddings = base_model(input_ids, attention_mask=attention_mask)[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return base_model(input_ids, attention_mask=attention_mask)[1]

    elif isinstance(base_model, DistilBertModel):
        return base_model(input_ids, attention_mask=attention_mask)[0][:, 0]

    elif isinstance(base_model, RobertaModel):
        return base_model(input_ids, attention_mask=attention_mask)[0][:, 0, :]

    else:
        raise Exception

    # if pooling_method == 'bert_pooling':
    #     return base_model(input_ids, attention_mask=attention_mask)[1]
    # elif pooling_method == 'distilbert_pooling':
    #     return base_model(input_ids, attention_mask=attention_mask)[0][:, 0]
    # elif pooling_method == 'sbert_pooling':
    #     token_embeddings = base_model(input_ids, attention_mask=attention_mask)[0]  # First element of model_output contains all token embeddings
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # elif pooling_method == 'roberta_pooling':
    #     return base_model(input_ids, attention_mask=attention_mask)[0][:, 0, :]
    # else:
    #     raise Exception

    # if isinstance(base_model, BertModel):
    #     return base_model(input_ids, attention_mask=attention_mask)[1]
    # elif isinstance(base_model, DistilBertModel):
    #     return base_model(input_ids, attention_mask=attention_mask)[0][:, 0]


# def get_base_model(sequence_classification_model, model_name=MODEL_NAME):
#     base_model_attribute_name = MODELS[model_name]['base_model_attribute_name']
#
#     return getattr(sequence_classification_model, base_model_attribute_name)
