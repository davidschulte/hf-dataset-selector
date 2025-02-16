from hfselect import compute_task_ranking, find_esm_repo_ids


def test_ranking(bert_model_name, imdb_dataset):
    esm_repo_ids = find_esm_repo_ids(model_name=bert_model_name)[:12]
    esm_repo_ids.append("fake_repo_ids_for_error_triggering")
    esm_repo_ids.append("bert-base-multilingual-uncased")
    esm_repo_ids += ["a" * i for i in range(200)]

    task_ranking = compute_task_ranking(
        imdb_dataset, model_name=bert_model_name, esm_repo_ids=esm_repo_ids
    )

    assert len(task_ranking) == 12
