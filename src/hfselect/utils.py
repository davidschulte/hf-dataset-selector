from tqdm.auto import tqdm
from huggingface_hub import HfApi
from collections import defaultdict
from typing import Union
import warnings
from .ESM import ESM
from .ESMConfig import ESMConfig, InvalidESMConfigException


def find_esm_repo_ids(model_name: str) -> list[str]:
    hf_api = HfApi()
    model_infos = hf_api.list_models(filter=["embedding_space_map", f"BaseLM:{model_name}"])

    return [model_info.id for model_info in model_infos]


def fetch_esms(
        repo_ids: list[str],
        return_failed_repo_ids=False
) -> Union[list["ESM"], tuple[list["ESM"], dict[str, str]]]:
    esms = []
    errors = defaultdict(list)
    with tqdm(repo_ids, desc="Fetching ESMs", unit="ESM") as pbar:
        for repo_id in pbar:
            try:
                esm = ESM.from_pretrained(repo_id)

                if not ESMConfig.from_esm(esm).is_valid:
                    raise InvalidESMConfigException

                esms.append(esm)

            except Exception as e:
                errors[type(e).__name__].append(repo_id)

    if len(errors) > 0:
        warning_message = format_warning_message(
            errors=errors,
            len_repo_ids=len(repo_ids),
            return_failed_repo_ids=return_failed_repo_ids
        )
        warnings.warn(warning_message, Warning)

    return esms if not return_failed_repo_ids else (esms, errors)


def fetch_esm_configs(
        repo_ids: list[str],
        return_failed_repo_ids=False
):
    esm_configs = []
    errors = defaultdict(list)
    with tqdm(repo_ids, desc="Fetching ESM Configs", unit="ESM Config") as pbar:
        for repo_id in pbar:
            try:
                esm_config = ESMConfig.from_pretrained(repo_id)

                if not esm_config.is_valid:
                    raise InvalidESMConfigException

                esm_configs.append(esm_config)
            except Exception as e:
                errors[type(e).__name__].append(repo_id)

    if len(errors) > 0:
        warning_message = format_warning_message(
            errors=errors,
            len_repo_ids=len(repo_ids),
            return_failed_repo_ids=return_failed_repo_ids
        )
        warnings.warn(warning_message, Warning)

    return esm_configs


def format_warning_message(errors: dict[str, str], len_repo_ids: int, return_failed_repo_ids: bool):
    warning_message = f"Fetching {len(errors)}/{len_repo_ids} failed. The following errors occurred:\n"
    warning_message += "\n".join([f"{error_name}: {len(error_repo_ids)}"
                                  for error_name, error_repo_ids in errors.items()])
    if not return_failed_repo_ids:
        warning_message += "\nTo return a dictionary of failed Repo IDs, set the 'return_failed_repo_ids' argument."

    return warning_message
