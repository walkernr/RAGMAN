import os
import gc
import argparse
from core.model_handler import ModelHandler
from core.corpus_handler import CorpusHandler, CorpusObject


def build_corpus(dataset):
    source_path = os.path.join("./documents", dataset)
    corpus_path = "./.corpus/{}".format(dataset)
    n_proc = 30
    batch_size = 10000
    model_handler = ModelHandler()
    corpus_handler = CorpusHandler()
    model_handler.load_corpus_model()
    corpus_model = model_handler.fetch_corpus_model(n_proc, batch_size)
    corpus_handler.make_corpus(
        source_path,
        corpus_path,
        corpus_model,
    )


if __name__ == "__main__":
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
        default="nq",
        choices=[
            "all",
            *datasets,
        ],
    )
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "all":
        for dataset in datasets:
            print("Building corpus for {}".format(dataset))
            build_corpus(dataset)
            gc.collect()
    else:
        print("Building corpus for {}".format(dataset))
        build_corpus(dataset)
