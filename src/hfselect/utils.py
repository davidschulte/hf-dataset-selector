from tqdm.auto import tqdm
from typing import Optional
from huggingface_hub import HfApi, model_info, ModelInfo
from collections import defaultdict
from .esm import ESM, ESMNotInitializedError
from .esmconfig import ESMConfig, InvalidESMConfigError
from hfselect import logger


def find_esm_repo_ids(
    model_name: Optional[str], filters: Optional[list[str]] = None
) -> list[str]:
    """
    Finds all ESM repo IDs for the specified language model name and filters

    Args:
        model_name: The name of the base language model
        filters: Filters for selecting ESMs (see hf_api.list_models)

    Returns:
        A list of ESM repo IDs
    """
    esm_infos = find_esm_model_infos(model_name, filters=filters)
    return [esm_info.id for esm_info in esm_infos]


def find_esm_model_infos(
    model_name: Optional[str], filters: Optional[list[str]] = None
) -> list[ModelInfo]:
    """
    Finds HF ModelInfos for all ESMs specified by the language model name and filters

    Args:
        model_name: The name of the base language model
        filters: Filters for selecting ESMs (see hf_api.list_models)

    Returns:
        A list of ESM ModelInfos
    """
    hf_api = HfApi()

    if filters is None:
        filters = []
    elif isinstance(filters, str):
        filters = [filters]

    filters.append("embedding_space_map")

    if model_name:
        # Make sure that the possibly redirected repo name is used,
        # e.g. google-bert/bert-base-uncased instead of bert-base-uncased
        model_name = model_info(model_name).id

        filters.append(f"base_model:{model_name}")

    return list(hf_api.list_models(filter=filters))


def fetch_esms(
    repo_ids: list[str],
) -> list[ESM]:
    """
    Fetches ESMs by their repo IDs. Invalid ESMs are excluded from the results. This can be seen in the logs.

    Args:
        repo_ids: The HF repo IDs of the ESMs

    Returns:
        A list of ESMs
    """
    esms = []
    errors = defaultdict(list)
    with tqdm(repo_ids, desc="Fetching ESMs", unit="ESM") as pbar:
        for repo_id in pbar:
            try:
                esm = ESM.from_pretrained(repo_id)
                esm.convert_legacy_to_new()

                if not esm.is_initialized:
                    raise ESMNotInitializedError

                if not esm.create_config().is_valid:
                    raise InvalidESMConfigError

                esms.append(esm)

            except Exception as e:
                errors[type(e).__name__].append(repo_id)

    if len(errors) > 0:
        if len(errors) > 0:
            logger.warning(
                f"Fetching  ESMs failed for {sum(map(len, errors.values()))} of {len(repo_ids)} repo IDs."
            )
            logger.debug(errors)

    return esms


def fetch_esm_configs(
    repo_ids: list[str],
) -> list[ESMConfig]:
    """
    Fetches ESMConfigs by their repo IDs. Invalid ESMConfigs are excluded from the results. This can be seen in the logs.

    Args:
        repo_ids: The HF repo IDs of the ESMs

    Returns:
        A list of ESMConfigs
    """
    esm_configs = []
    errors = defaultdict(list)
    with tqdm(repo_ids, desc="Fetching ESM Configs", unit="ESM Config") as pbar:
        for repo_id in pbar:
            try:
                esm_config = ESMConfig.from_pretrained(repo_id)

                if not esm_config.is_valid:
                    raise InvalidESMConfigError

                esm_configs.append(esm_config)
            except Exception as e:
                errors[type(e).__name__].append(repo_id)

    if len(errors) > 0:
        logger.warning(
            f"Fetching ESM configs failed for {sum(map(len, errors.values()))} of {len(repo_ids)} repo IDs."
        )
        logger.debug(errors)

    return esm_configs
