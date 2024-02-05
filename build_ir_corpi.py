import os
import gc
import argparse
from core.model_handler import ModelHandler
from core.corpus_handler import CorpusHandler, CorpusObject


def build_corpus(dataset, n_proc, batch_size):
    source_path = os.path.join("./documents", dataset)
    corpus_path = "./.corpus/{}".format(dataset)
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
        default="scifact",
        choices=[
            "all",
            *datasets,
        ],
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
    )
    args = parser.parse_args()
    dataset, n_proc, batch_size = args.dataset, args.n_proc, args.batch_size
    if dataset == "all":
        for dataset in datasets:
            print("Building corpus for {}".format(dataset))
            build_corpus(dataset, n_proc, batch_size)
            gc.collect()
    else:
        print("Building corpus for {}".format(dataset))
        build_corpus(dataset, n_proc, batch_size)
