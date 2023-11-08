import os
import ujson as json
import numpy as np
import scipy.stats as st
from tqdm import tqdm
import argparse
import random
from ragman import RAGMAN


def f_beta(p, r, beta):
    if p == 0 or r == 0:
        return 0
    else:
        return (1 + beta**2) * p * r / (beta**2 * p + r)


def get_sorted_order(ground_truths, scores, predicted):
    custom_sort_scores = [
        -1e5 if item not in predicted else predicted.index(item)
        for item in ground_truths
    ]
    score_order = np.lexsort((custom_sort_scores, -1 * np.array(scores)))
    return score_order


def score_retrieval(
    predicted,
    ground_truths,
    total,
    scores,
):
    score_order = get_sorted_order(ground_truths, scores, predicted)
    scores = np.array(scores)[score_order]
    ground_truths = np.array(ground_truths)[score_order]
    t = total
    pp = len(predicted)
    pn = t - pp
    p = len(ground_truths)
    n = t - p
    tp = len(set(predicted).intersection(set(ground_truths)))
    fp = pp - tp
    fn = p - tp
    tn = t - tp - fp - fn

    prev = p / t
    acc = (tp + tn) / t
    recall = tp / p
    tnr = tn / n
    fpr = fp / n
    fnr = fn / p
    inf = recall + tnr - 1
    pt = 0 if recall - fpr == 0 else (np.sqrt(recall * fpr) - fpr) / (recall - fpr)
    balacc = (recall + tnr) / 2
    precision = tp / pp if pp > 0 else 0
    fdr = fp / pp if pp > 0 else 0
    fomr = fn / pn if pn > 0 else 0
    npv = tn / pn if pn > 0 else 0
    plr = 0 if fpr == 0 else recall / fpr
    nlr = 0 if tnr == 0 else fnr / tnr
    mrkdns = precision + npv - 1
    dor = 0 if nlr == 0 else plr / nlr
    f1 = f_beta(precision, recall, 1)
    f2 = f_beta(precision, recall, 2)
    f3 = f_beta(precision, recall, 3)
    f4 = f_beta(precision, recall, 4)
    fmi = np.sqrt(precision * recall)
    mcc = np.sqrt(recall * tnr * precision * npv) - np.sqrt(fnr * fpr * fomr * fdr)
    ji = tp / (tp + fn + fp)

    idcg = np.sum([score / np.log2(i + 2) for i, score in enumerate(scores)])
    if tp > 0:
        dcg = 0
        for score, ground_truth in zip(scores, ground_truths):
            if ground_truth in predicted:
                position = predicted.index(ground_truth)
                dcg += score / np.log2(position + 2)
    else:
        dcg = 0
    ndcg = dcg / idcg

    limit = min(len(predicted), 10)
    idcg10 = np.sum([score / np.log2(i + 2) for i, score in enumerate(scores[:limit])])
    if tp > 0:
        dcg10 = 0
        for score, ground_truth in zip(scores, ground_truths):
            if ground_truth in predicted[:limit]:
                position = predicted[:limit].index(ground_truth)
                dcg10 += score / np.log2(position + 2)
    else:
        dcg10 = 0
    ndcg10 = dcg10 / idcg10

    score_dict = {
        "positive": p,
        "negative": n,
        "predicted_positive": pp,
        "predicted_negative": pn,
        "prevalence": prev,
        "accuracy": acc,
        "balanced_accuracy": balacc,
        "precision": precision,
        "false_discovery_rate": fdr,
        "false_omission_rate": fomr,
        "negative_predictive_value": npv,
        "markedness": mrkdns,
        "recall": recall,
        "specificity": tnr,
        "fall-out": fpr,
        "miss_rate": fnr,
        "informedness": inf,
        "prevalence_threshold": pt,
        "positive_likelihood_ratio": plr,
        "negative_likelihood_ratio": nlr,
        "diagnostic_odds_ratio": dor,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "fowlkes-mallows_index": fmi,
        "matthews_correlation_coefficient": mcc,
        "jaccard_index": ji,
        "normalized_discounted_cumulative_gain": ndcg,
        "normalized_discounted_cumulative_gain_{}".format(limit): ndcg10,
    }
    return score_dict


class TEST:
    def __init__(
        self,
        retrieval_config=[{"name": "vector", "parameters": {"k": 5}}],
        embedding_model_path="BAAI/llm-embedder",
        cross_encoding_model_path="BAAI/bge-reranker-large",
        query_model_path="TheBloke/Mistral-7B-OpenOrca-GPTQ",
        validate_retrieval=False,
        n_proc=16,
        corpus_processing_batch_size=10000,
        corpus_encoding_batch_size=512,
        ranking_batch_size=256,
        max_tokens=4096,
        max_new_tokens=512,
        retrieval_device="cuda:0",
        query_device=None,
    ):
        self.ragman = RAGMAN(**{k: v for k, v in locals().items() if k != "self"})
        self.queries = {}

    def update_retrieval_config(self, retrieval_config):
        self.ragman.update_retrieval_config(retrieval_config)

    def get_retrieval_config(self):
        return self.ragman.retrieval_config

    def load_corpus(self, dataset):
        try:
            self.ragman.load_corpus(dataset)
        except:
            self.ragman.build_corpus(dataset)
            self.ragman.load_corpus(dataset)

    def reload_models(self, embedding_model_path, cross_encoding_model_path):
        self.ragman.embedding_model_path = embedding_model_path
        self.ragman.cross_encoding_model_path = cross_encoding_model_path
        self.ragman.load_models()

    def load_queries(self, dataset):
        query_path = "./queries/{}/queries_test.json".format(dataset)
        with open(query_path, "r") as f:
            self.queries[dataset] = json.load(f)
        # if len(self.queries[dataset]) > 100:
        #     self.queries[dataset] = random.sample(self.queries[dataset], 100)

    def run_retrieval_experiment(self, dataset):
        reports = []
        for query in tqdm(self.queries[dataset], desc="Running Retrieval Experiment"):
            passages = self.ragman.search(dataset, query["query"])
            report = score_retrieval(
                [p.passage_id for p in passages.get_passages()],
                query["offset"],
                len(self.ragman.corpus[dataset].corpus_doc_index) - 1,
                query["score"],
            )
            reports.append({"query": query["query"], "report": report})
        return reports

    def collate_reports(self, reports):
        report = {}
        for metric in reports[0]["report"].keys():
            v = np.array([r["report"][metric] for r in reports])
            n = len(v)
            m = np.mean(v)
            ci = st.t.interval(confidence=0.95, df=n - 1, loc=m, scale=st.sem(v))
            report[metric] = {
                "support": n,
                "mean": float(m),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "q1": float(np.quantile(v, 0.25)),
                "q3": float(np.quantile(v, 0.75)),
                "c0": float(ci[0]),
                "c1": float(ci[1]),
            }
        return report

    def test(self, dataset):
        if dataset in self.ragman.corpus:
            if dataset in self.queries:
                reports = self.run_retrieval_experiment(dataset)
            else:
                self.load_queries(dataset)
                reports = self.run_retrieval_experiment(dataset)
        else:
            if dataset in self.queries:
                self.load_corpus(dataset)
                reports = self.run_retrieval_experiment(dataset)
            else:
                self.load_corpus(dataset)
                self.load_queries(dataset)
                reports = self.run_retrieval_experiment(dataset)
        collated_report = self.collate_reports(reports)
        model_dict = {}
        names = [config["name"] for config in self.ragman.retrieval_config]
        if "bm25" in names:
            model_dict["lexical"] = "BM25+"
        if "vector" in names:
            model_dict["embedding"] = self.ragman.embedding_model_path
        if "cross_encoder" in names:
            model_dict["cross_encoding"] = self.ragman.cross_encoding_model_path
        return {
            "dataset": dataset,
            "models": model_dict,
            "retrieval_config": self.get_retrieval_config(),
            "collated_report": collated_report,
            "reports": reports,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="nfcorpus",
        choices=[
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
        ],
    )
    embedding_model_paths = [
        "BAAI/llm-embedder",
        "BAAI/bge-large-en-v1.5",
    ]
    cross_encoding_model_paths = [
        "BAAI/bge-reranker-large",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ]
    args = parser.parse_args()
    dataset = args.dataset
    retrieval_configs = {
        "bm25-0": [
            {
                "name": "bm25",
                "parameters": {
                    "keyword_k": 0,
                    "k": 100,
                },
            }
        ],
        "bm25-09": [
            {
                "name": "bm25",
                "parameters": {
                    "keyword_k": 0.9,
                    "k": 100,
                },
            }
        ],
        # "vctr": [
        #     {
        #         "name": "vector",
        #         "parameters": {
        #             "k": 100,
        #         },
        #     }
        # ],
        # "pool-50-50-bm25-0-vctr": [
        #     {
        #         "name": "pool",
        #         "parameters": {
        #             "k": 100,
        #             "retriever_config": [
        #                 {
        #                     "name": "bm25",
        #                     "parameters": {
        #                         "keyword_k": 0,
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.5,
        #                 },
        #                 {
        #                     "name": "vector",
        #                     "parameters": {
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.5,
        #                 },
        #             ],
        #         },
        #     }
        # ],
        # "pool-50-50-bm25-1-vctr": [
        #     {
        #         "name": "pool",
        #         "parameters": {
        #             "k": 100,
        #             "retriever_config": [
        #                 {
        #                     "name": "bm25",
        #                     "parameters": {
        #                         "keyword_k": 1,
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.5,
        #                 },
        #                 {
        #                     "name": "vector",
        #                     "parameters": {
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.5,
        #                 },
        #             ],
        #         },
        #     }
        # ],
        # "pool-75-25-bm25-0-vctr": [
        #     {
        #         "name": "pool",
        #         "parameters": {
        #             "k": 100,
        #             "retriever_config": [
        #                 {
        #                     "name": "bm25",
        #                     "parameters": {
        #                         "keyword_k": 0,
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.75,
        #                 },
        #                 {
        #                     "name": "vector",
        #                     "parameters": {
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.25,
        #                 },
        #             ],
        #         },
        #     }
        # ],
        # "pool-75-25-bm25-1-vctr": [
        #     {
        #         "name": "pool",
        #         "parameters": {
        #             "k": 100,
        #             "retriever_config": [
        #                 {
        #                     "name": "bm25",
        #                     "parameters": {
        #                         "keyword_k": 1,
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.75,
        #                 },
        #                 {
        #                     "name": "vector",
        #                     "parameters": {
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.25,
        #                 },
        #             ],
        #         },
        #     }
        # ],
        # "pool-25-75-bm25-0-vctr": [
        #     {
        #         "name": "pool",
        #         "parameters": {
        #             "k": 100,
        #             "retriever_config": [
        #                 {
        #                     "name": "bm25",
        #                     "parameters": {
        #                         "keyword_k": 0,
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.25,
        #                 },
        #                 {
        #                     "name": "vector",
        #                     "parameters": {
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.75,
        #                 },
        #             ],
        #         },
        #     }
        # ],
        # "pool-25-75-bm25-1-vctr": [
        #     {
        #         "name": "pool",
        #         "parameters": {
        #             "k": 100,
        #             "retriever_config": [
        #                 {
        #                     "name": "bm25",
        #                     "parameters": {
        #                         "keyword_k": 1,
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.25,
        #                 },
        #                 {
        #                     "name": "vector",
        #                     "parameters": {
        #                         "k": 1000,
        #                     },
        #                     "weight": 0.75,
        #                 },
        #             ],
        #         },
        #     }
        # ],
        # "bm25-0-ce": [
        #     {
        #         "name": "bm25",
        #         "parameters": {
        #             "keyword_k": 0,
        #             "k": 100,
        #         },
        #     },
        #     {
        #         "name": "cross_encoder",
        #         "parameters": {
        #             "passage_search": True,
        #             "k": 100,
        #         },
        #     },
        # ],
        # "bm25-1-ce": [
        #     {
        #         "name": "bm25",
        #         "parameters": {
        #             "keyword_k": 1,
        #             "k": 100,
        #         },
        #     },
        #     {
        #         "name": "cross_encoder",
        #         "parameters": {
        #             "passage_search": True,
        #             "k": 100,
        #         },
        #     },
        # ],
        # "vctr-ce": [
        #     {
        #         "name": "vector",
        #         "parameters": {
        #             "k": 100,
        #         },
        #     },
        #     {
        #         "name": "cross_encoder",
        #         "parameters": {
        #             "passage_search": True,
        #             "k": 100,
        #         },
        #     },
        # ],
    }
    if not os.path.exists("./reports"):
        os.mkdir("./reports")
    session = TEST()
    for embedding_model_path in embedding_model_paths:
        for cross_encoding_model_path in cross_encoding_model_paths:
            session.reload_models(embedding_model_path, cross_encoding_model_path)
            for test_name, retrieval_config in retrieval_configs.items():
                model_name = []
                if "bm25" in test_name:
                    model_name.append("BM25")
                if "vctr" in test_name:
                    model_name.append(embedding_model_path.split("/")[-1])
                if "ce" in test_name:
                    model_name.append(cross_encoding_model_path.split("/")[-1])
                _id = "_".join([dataset, *model_name, test_name])
                report_name = "./reports/report_{}.json".format(_id)
                if not os.path.exists(report_name):
                    session.update_retrieval_config(retrieval_config)
                    report = session.test(dataset)
                    with open(report_name, "w") as f:
                        json.dump(report, f, indent=2)
                    print("Report: {}".format(test_name))
                    print(json.dumps(report["collated_report"], indent=2))
