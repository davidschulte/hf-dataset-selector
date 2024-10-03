import os

import torch
import numpy as np
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

CLASSIFICATION_TARGETS_TASKS = [
    'imdb_plain_text',
    'tweet_eval_emotion',
    'tweet_eval_sentiment',
    'paws-x_en',
    'md_gender_bias_convai2_inferred'
]

REGRESSION_TARGET_TASKS = [
    'llm-book__JGLUE_JSTS',
    'google_wellformed_query_default',
    'google__civil_comments_default'
]

TARGET_TASKS = CLASSIFICATION_TARGETS_TASKS + REGRESSION_TARGET_TASKS

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


MODEL_NAME = 'bert_multilang'
TEXT_SEPARATOR = ' [SEP] '

ESM_OPTIONAL_LAYER_DIMS = None

GPU_DEVICE = 'cuda'

MODELS_BASE_DIR = os.path.join('models', MODEL_NAME)
MODELS_SINGLE_DIR = os.path.join(MODELS_BASE_DIR, 'single')
MODELS_TRANSFER_DIR = os.path.join(MODELS_BASE_DIR, 'transfers')
MODELS_FROZEN_TRANSFER_DIR = os.path.join(MODELS_BASE_DIR, 'frozen_transfers')
MODELS_SOURCES_DIR = os.path.join(MODELS_BASE_DIR, 'sources')
MODELS_SOURCES_BIN_DIR = os.path.join(MODELS_BASE_DIR, 'sources_bin')
EVAL_BASE_DIR = os.path.join('eval', MODEL_NAME)
EVAL_TRANSFER_DIR = os.path.join(EVAL_BASE_DIR, 'transfers')
EVAL_FROZEN_TRANSFER_DIR = os.path.join(EVAL_BASE_DIR, 'frozen_transfers')
TRANSFORMATION_NETS_FILEPATH = os.path.join(MODELS_BASE_DIR, 'transformation_networks')
DATASETS_DIR = 'datasets'
EMBEDDINGS_BASE_DIR = os.path.join('embeddings', MODEL_NAME)
EMBEDDING_SPACE_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_BASE_DIR, 'embedding_space')
LOGME_DIR = os.path.join(EVAL_BASE_DIR, 'logme')
LOGME_TNN_DIR = os.path.join(EVAL_BASE_DIR, 'logme_tnn')
SIMPLE_LOGME_TNN_DIR = os.path.join(EVAL_BASE_DIR, 'simple_logme_tnn')
LEEP_DIR = os.path.join(EVAL_BASE_DIR, 'leep')
NCE_DIR = os.path.join(EVAL_BASE_DIR, 'nce')
VOCAB_OVERLAP_DIR = os.path.join(EVAL_BASE_DIR, 'vocab_overlap')
TEXTEMB_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_BASE_DIR, 'textemb')
TEXTEMB_EVAL_DIR = os.path.join(EVAL_BASE_DIR, 'textemb')
TASKEMB_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_BASE_DIR, 'taskemb')
TASKEMB_EVAL_DIR = os.path.join(EVAL_BASE_DIR, 'taskemb')
TASKEMB_FROZEN_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_BASE_DIR, 'taskemb_frozen')
TASKEMB_FROZEN_EVAL_DIR = os.path.join(EVAL_BASE_DIR, 'taskemb_frozen')


LOGGING_DIR = 'logs'
