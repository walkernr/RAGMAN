import os
import gc
import argparse
import ujson as json
import datasets as ds

# import ir_datasets
from tqdm import tqdm


def build_documents(dataset, beir=True):
    if beir:
        data_path = "BeIR/{}".format(dataset)
        data_name = data_path.split("/")[1]
        corpus = ds.load_dataset(data_path, "corpus")["corpus"]
        queries = ds.load_dataset(data_path, "queries")["queries"]
        rels = {}
        for split in ["train", "validation", "test"]:
            try:
                rels[split] = ds.load_dataset(data_path + "-qrels", split)[split]
            except:
                pass
    else:
        if dataset == "sara":
            data_name = "sara"
            data = ir_datasets.load(dataset)
            corpus = data.docs_iter()
            queries = data.queries_iter()
            rels = data.qrels_iter()
        else:
            data_path = "irds/{}".format(dataset)
            data_name = data_path.split("/")[1]
            corpus = ds.load_dataset(data_path, "docs")
            if dataset == "nfcorpus":
                test_path = "{}_test".format(data_path)
                queries = ds.load_dataset(test_path, "queries")
                rels = ds.load_dataset(test_path, "qrels")
            else:
                queries = ds.load_dataset(data_path, "queries")
                rels = ds.load_dataset(data_path, "qrels")

    if dataset == "sara":
        corpus_id_text = [
            (x.doc_id, x.text) for x in tqdm(corpus, desc="Collecting Contexts")
        ]
        c_id = {x[0]: i for i, x in enumerate(corpus_id_text)}
        corpus_text = [x[1] for x in corpus_id_text]
    else:
        corpus_text = [
            x["text"] if dataset != "sara" else x.text
            for x in tqdm(corpus, desc="Collecting Contexts")
        ]

        c_id = {
            str(x["_id" if beir else "doc_id"] if dataset != "sara" else x.doc_id): i
            for i, x in tqdm(enumerate(corpus), desc="Collecting Context Indices")
        }

    if beir:
        q_id = {
            str(x["_id"]): i
            for i, x in tqdm(enumerate(queries), desc="Collecting Query Indices")
        }

        query_data = {
            split: [
                {
                    "query": x["text"],
                    "response": "",
                    "reference": [],
                    "offset": [],
                    "score": [],
                }
                for x in queries
            ]
            for split in rels.keys()
        }
        for split in rels.keys():
            for x in tqdm(
                rels[split], desc="Mapping Queries to Contexts ({})".format(split)
            ):
                try:
                    u = q_id[str(x["query-id"])]
                    v = c_id[str(x["corpus-id"])]
                    w = x["score"]
                    if w > 0:
                        query_data[split][u]["reference"].append(
                            "{}_passages.json".format(data_name)
                        )
                        query_data[split][u]["offset"].append(v)
                        query_data[split][u]["score"].append(w)
                except:
                    pass
    else:
        query_data = {
            "test": {
                (q["query_id"] if dataset != "sara" else q.query_id): {
                    "query": (q["text"] if dataset != "sara" else q.text),
                    "response": "",
                    "reference": [],
                    "offset": [],
                    "score": [],
                }
                for q in queries
            }
        }
        for split in query_data:
            for rel in rels:
                u = rel["query_id"] if dataset != "sara" else rel.query_id
                if u in query_data[split].keys():
                    v = c_id[str(rel["doc_id"] if dataset != "sara" else rel.doc_id)]
                    w = rel["relevance"] if dataset != "sara" else rel.relevance
                    if w > 0:
                        query_data[split][u]["reference"].append(
                            "{}_passages.json".format(data_name)
                        )
                        query_data[split][u]["offset"].append(v)
                        query_data[split][u]["score"].append(w)
        query_data = {split: list(query_data[split].values()) for split in query_data}
    document_save_path = os.path.join("./documents", data_name)
    if not os.path.exists(document_save_path):
        os.makedirs(document_save_path)

    with open(os.path.join(document_save_path, "passages.json"), "w") as f:
        f.write(json.dumps(corpus_text, indent=2))

    query_save_path = os.path.join("./queries", data_name)
    if not os.path.exists(query_save_path):
        os.makedirs(query_save_path)
    for split in query_data.keys():
        with open(
            os.path.join(query_save_path, "queries_{}.json".format(split)), "w"
        ) as f:
            f.write(
                json.dumps(
                    [q for q in query_data[split] if len(q["offset"]) > 0], indent=2
                )
            )


if __name__ == "__main__":
    beir_datasets = [
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
    # irds_datasets = [
    #     "cranfield",
    #     "sara",
    #     "vaswani",
    #     "highwire_trec-genomics-2006",
    # ]
    irds_datasets = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        choices=[
            "all",
            *beir_datasets,
            *irds_datasets,
        ],
    )
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "all":
        for dataset in beir_datasets + irds_datasets:
            print("Building corpus for {}".format(dataset))
            build_documents(dataset, beir=dataset in beir_datasets)
            gc.collect()
    else:
        beir = dataset in beir_datasets
        print("Building corpus for {}".format(dataset))
        build_documents(dataset, beir=beir)
