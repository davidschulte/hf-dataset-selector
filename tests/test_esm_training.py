from hfselect import ESMTrainer


def test_esm_training(bert_model, bert_tokenizer, imdb_dataset):
    trainer = ESMTrainer()

    trainer.train_with_models(
        dataset=imdb_dataset,
        base_model=bert_model,
        tuned_model=bert_model,
        tokenizer=bert_tokenizer,
    )
