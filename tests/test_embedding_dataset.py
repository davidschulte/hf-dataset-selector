from hfselect.embedding_dataset import create_embedding_dataset


def test_embedding_dataset(bert_model, bert_tokenizer, imdb_dataset):
    embedding_dataset = create_embedding_dataset(
        imdb_dataset, bert_model, bert_model, bert_tokenizer
    )

    assert len(embedding_dataset) == len(imdb_dataset)

    filepath = "./test_embedding_dataset.npz"
    embedding_dataset.save(filepath)
    embedding_dataset = embedding_dataset.from_disk(filepath)

    assert len(embedding_dataset) == len(imdb_dataset)
