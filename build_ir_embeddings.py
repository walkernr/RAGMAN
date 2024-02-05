import gc
import argparse
from core.model_handler import ModelHandler
from core.corpus_handler import CorpusHandler, CorpusObject


def embed_corpus(dataset, model_path, batch_size):
    corpus = CorpusObject()

    corpus_handler = CorpusHandler()
    model_handler = ModelHandler()
    model_handler.load_corpus_model()
    model_handler.load_embedding_model(model_path, "cuda:0")
    corpus_model = model_handler.fetch_corpus_model(1, 10000)
    embedding_model = model_handler.fetch_embedding_model(batch_size)
    corpus.load_corpus("./.corpus/{}".format(dataset), corpus_model)
    corpus_handler.make_embedding(
        "./.corpus/{}".format(dataset),
        corpus,
        embedding_model,
        corpus_model,
    )


if __name__ == "__main__":
    model_paths = [
        "BAAI/llm-embedder",
        "BAAI/bge-large-en-v1.5",
    ]
    datasets = [
        "msmarco",
        "trec-covid",
        "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "dbpedia-entity",
        "scidocs",
        "fever",
        "climate-fever",
        "scifact",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        choices=[
            "all",
            *datasets,
        ],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    args = parser.parse_args()
    dataset, batch_size = args.dataset, args.batch_size
    for model_path in model_paths:
        if dataset == "all":
            for dataset in datasets:
                print("Embedding corpus for {} with {}".format(dataset, model_path))
                embed_corpus(dataset, model_path, batch_size)
                gc.collect()
        else:
            print("Embedding corpus for {} with {}".format(dataset, model_path))
            embed_corpus(dataset, model_path, batch_size)
