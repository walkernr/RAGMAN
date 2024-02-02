import numpy as np
import faiss
from tqdm import tqdm
from core.model_handler import (
    clean_text,
    constant_thresholding,
    gap_thresholding,
    elbow_thresholding,
)

#############################################################
#### This module handles everything related to retrieval ####
#############################################################


def build_retriever(config):
    """
    Builds a retriever according to a provided condfiguration
    input: config (dict)
    output: retriever (Retriever)
    """
    if config["name"] == "bm25":
        return BM25Retriever(**config["parameters"])
    elif config["name"] == "graph":
        return GraphRetriever(**config["parameters"])
    elif config["name"] == "vector":
        return VectorRetriever(**config["parameters"])
    elif config["name"] == "cross_encoder":
        return CrossEncoderRetriever(**config["parameters"])
    elif config["name"] == "pool":
        # pool retriever is a special case that builds a retriever from a list of retrievers
        return PoolRetriever(config["parameters"])


class BM25Retriever:
    """
    a retriever using the BM25 algorithm
    lexical-based retrieval
    uses only lemmas with no stop words or punctuation
    """

    def __init__(
        self,
        keyword_k,
        k,
    ):
        """
        initializer
        input: keyword_k (int)
               k (int)
        """
        # set name and mode
        self.name = "bm25"
        self.mode = "passage"
        # set search parameters
        # keyword k for keyword expansion
        # k for number of passages to return
        self.keyword_k = keyword_k
        self.k = k

    def retrieve(
        self,
        corpus,
        query,
        prior_scores,
        prior_hits,
        model,
    ):
        """
        retrieves passages from corpus using BM25
        input: corpus (Corpus)
               query (str)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               model (CorpusModel)
        output: scores (np.ndarray)
                hits (np.ndarray)
        """
        # extract and expand keywords
        keywords = model.extract_keywords(query)
        keywords = model.get_similar_keywords(corpus, keywords, self.keyword_k)
        # if the top-k is not zero and keywords are found
        if self.k != 0 and len(keywords) > 0:
            print("BM25: Ranking Passages [processing]")
            # adjust k for number of prior hits
            adj_k = min(len(prior_hits), self.k)
            # retrieve scores and hits
            passage_scores = corpus.passage_bm25_model.get_scores(keywords)
            passage_scores = np.array([passage_scores[i] for i in prior_hits])
            hits = np.argsort(passage_scores)[::-1]
            # get sorted scores and hits
            sorted_scores = passage_scores[hits]
            # if k is negative, threshold
            if self.k < 0:
                # either gap or elbow thresholding
                if self.k == -1:
                    thresh_k = gap_thresholding(sorted_scores)
                elif self.k == -2:
                    thresh_k = elbow_thresholding(sorted_scores)
                # create mask
                explicit_mask = np.zeros(len(sorted_scores))
                explicit_mask[:thresh_k] = 1
            else:
                # else the mask just goes up to k
                explicit_mask = np.zeros(len(sorted_scores))
                explicit_mask[:adj_k] = 1
            # apply mask
            scores = sorted_scores[explicit_mask == 1]
            hits = hits[explicit_mask == 1]
            return scores, hits
        elif len(keywords) == 0:
            print("BM25: No keywords found [skipping]")
            scores, hits = prior_scores, prior_hits
        else:
            print("BM25: Top-k is zero [skipping]")
            scores, hits = prior_scores, prior_hits
        return scores, hits


class GraphRetriever:
    """
    a retriever based on a signed similarity graph
    relies on prior hits, as the results are not conditioned on the query
    prior scores used for initial ranking
    ranking updated based on graph-based passage similarity
    the "hubs" are returned
    """

    def __init__(
        self,
        k,
    ):
        """
        initializer
        input: k (int)
        """
        # set name and mode
        self.name = "graph"
        self.mode = "passage"
        # set convergence tolerance and k
        self.tol = 1e-6
        self.k = k
        self.sim_thresh = 0.9
        self.perc_thresh = 0.9

    def expand_hits(self, corpus, prior_hits):
        """
        transforms prior hits into a set of expanded hits based on graph-based similarity
        selects new hits based on >= constant similarity threshold and >= 90th percentile of similarities
        input: corpus (Corpus)
               prior_hits (np.ndarray)
               sim_thresh (float)
               perc_thresh (float)
        output: hits (np.ndarray)
        """
        # initialize inner product index
        index = faiss.IndexFlatIP(corpus.passage_embeddings.shape[1])
        index.add(corpus.passage_embeddings[prior_hits])
        # retrieve similarities
        all_hits_scores = {}
        for hit in prior_hits:
            scores, hits = index.search(
                corpus.passage_embeddings[hit].reshape(1, -1), 10
            )
            scores, hits = scores[0], hits[0]
            for s, h in zip(scores, hits):
                # ignore self-hits
                if h != hit:
                    # only take the highest similarity score for a hit
                    if h in all_hits_scores.keys():
                        if s > all_hits_scores[h]:
                            all_hits_scores[h] = s
                    else:
                        all_hits_scores[h] = s
        # get percentile threshold for similarity
        sim_perc_thresh = np.quantile(list(all_hits_scores.values()), self.perc_thresh)
        # prune hits
        hits = np.array(
            [
                h
                for h, s in all_hits_scores.items()
                if s >= sim_perc_thresh and s >= self.sim_thresh
            ]
        ).astype(int)
        hits = set(hits)
        # union with prior hits
        hits = hits.union(set(prior_hits))
        hits = np.array(list(hits))
        return hits

    def retrieve(self, corpus, query, prior_scores, prior_hits, model):
        """
        retrieves passages from corpus using graph-based similarity
        note that the query and model arguments are not used, but are included to ensure consistency with other retrievers
        input: corpus (Corpus)
               query (str)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               model (None)
        output: scores (np.ndarray)
                hits (np.ndarray)
        """
        # if the top-k is not zero
        if self.k != 0:
            print("Graph: Ranking Passages [processing]")
            # adjust k for number of prior hits
            adj_k = min(len(prior_hits), self.k)
            # expand hits based on similarity
            expanded_hits = self.expand_hits(corpus, prior_hits)
            # create similarity matrix
            similarity_matrix = np.dot(
                corpus.passage_embeddings[expanded_hits],
                corpus.passage_embeddings[expanded_hits].T,
            )
            # index map to find similarity matrix indices from hit indices
            expanded_hits_dict = {h: i for i, h in enumerate(expanded_hits)}
            # initialize scores with prior scores (new passages will be zero to start)
            scores = np.zeros(len(expanded_hits))
            for i, (score, hit) in enumerate(zip(prior_scores, prior_hits)):
                if hit in expanded_hits_dict:
                    scores[expanded_hits_dict[hit]] = score
            # normalize scores
            scores /= np.sum(scores)
            # initialize previous scores for convergence condition
            prev_scores = np.zeros(len(expanded_hits))
            # count iterations
            i = 0
            # while the convergence criterion is not met
            while np.linalg.norm(scores - prev_scores, 2) > self.tol:
                # update previous scores
                prev_scores = scores.copy()
                # update scores
                scores = np.dot(similarity_matrix, scores)
                # normalize scores
                scores /= np.linalg.norm(scores, 2)
                i += 1
            print("Converged in {} iterations".format(i))
            # get sorted scores and hits
            score_order = np.argsort(scores)[::-1]
            sorted_scores = scores[score_order]
            hits = expanded_hits[score_order]
            # if k is negative, threshold
            if self.k < 0:
                # either gap, elbow, or constant thresholding
                if self.k == -1:
                    thresh_k = gap_thresholding(sorted_scores)
                elif self.k == -2:
                    thresh_k = elbow_thresholding(sorted_scores)
                else:
                    thresh_k = constant_thresholding(sorted_scores, self.k)
                # create mask
                explicit_mask = np.zeros(len(sorted_scores))
                explicit_mask[:thresh_k] = 1
            else:
                # else the mask just goes up to k
                explicit_mask = np.zeros(len(sorted_scores))
                explicit_mask[:adj_k] = 1
            # apply mask
            scores = sorted_scores[explicit_mask == 1]
            hits = hits[explicit_mask == 1]
        else:
            scores, hits = prior_scores, prior_hits
        return scores, hits


class VectorRetriever:
    """
    a retriever based on vector similarity between a query embedding and passage embeddings in a vector store
    """

    def __init__(
        self,
        k,
    ):
        """
        initializer
        input: k (int)
        """
        # set name and mode
        self.name = "vector"
        self.mode = "passage"
        # set k
        self.k = k

    def retrieve(
        self,
        corpus,
        query,
        prior_scores,
        prior_hits,
        model,
    ):
        """
        retrieves passages from corpus using vector similarity
        input: corpus (Corpus)
               query (str)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               model (EmbeddingModel)
        output: scores (np.ndarray)
                hits (np.ndarray)
        """
        # if the top-k is not zero
        if self.k != 0:
            # embed query
            query_embedding = model.encode_query(query)
            print("Vector: Ranking Passages [processing]")
            n_prior_hits = len(prior_hits)
            # adjust k for number of prior hits
            adj_k = min(n_prior_hits, self.k)
            # initialize inner product index
            index = faiss.IndexFlatIP(corpus.passage_embeddings.shape[1])
            index.add(corpus.passage_embeddings[prior_hits])
            # if k is negative, threshold
            if self.k < 1:
                # get the full search
                scores, hits = index.search(query_embedding, n_prior_hits)
                scores, hits = scores[0], hits[0]
                # gap or elbow thresholding
                if self.k == -1:
                    thresh_k = gap_thresholding(scores)
                elif self.k == -2:
                    thresh_k = elbow_thresholding(scores)
                else:
                    thresh_k = constant_thresholding(scores, self.k)
                # create mask
                mask = np.zeros(len(scores))
                mask[:thresh_k] = 1
                # apply mask
                scores = scores[mask == 1]
                hits = prior_hits[hits[mask == 1]]
            else:
                # else the mask just goes up to k
                scores, hits = index.search(query_embedding, adj_k)
                scores, hits = scores[0], hits[0]
                hits = prior_hits[hits]
        else:
            scores, hits = prior_scores, prior_hits
        return scores, hits


class CrossEncoderRetriever:
    """
    a retriever based on cross-encoder similarity between a query and sentences
    """

    def __init__(
        self,
        passage_search,
        pooling,
        k,
    ):
        """
        initializer
        input: passage_search (bool)
               k (int)
        """
        # set name and mode
        self.name = "cross_encoder"
        self.mode = "sentence"
        # set passage_search and k
        # if passage_search is true, the top-k passages are retrieved, otherwise the top-k sentences are returned
        self.passage_search = passage_search
        self.pooling = pooling
        self.k = k

    def retrieve(
        self,
        corpus,
        query,
        prior_scores,
        prior_hits,
        model,
    ):
        """
        retrieves sentences from corpus using cross-encoder similarity
        note that the model input needs to be a dictionary with both a CorpusModel and a CrossEncoderModel
        input: corpus (Corpus)
               query (str)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               model (dict[str:[CorpusModel, CrossEncoderModel]])
        output: scores (np.ndarray)
                hits (np.ndarray)
        """
        # if the top-k is not zero
        if self.k != 0:
            print("Cross-Encoder: Ranking Sentences [processing]")
            # get number of prior hits in terms of sentences and passages
            n_prior_hits = len(prior_hits)
            n_prior_passages = len(
                list(set([corpus.sentence_passage_id[j, 0] for j in prior_hits]))
            )
            # get the sentence scores
            scores = model["cross_encoder"].cross_encode_query_with_sentences(
                corpus, query, prior_hits, model["corpus"]
            )
            # get the hits and sorted scores
            hits = np.argsort(scores)[::-1]
            sorted_scores = scores[hits]
            # if k is negative, threshold
            if self.k < 1:
                # if passage_search, then the top-k passages are retrieved
                if self.passage_search:
                    # initialize passage score dict
                    passage_score_dict = {}
                    for score, hit in zip(sorted_scores, prior_hits[hits]):
                        passage = corpus.sentence_passage_id[hit, 0]
                        if passage in passage_score_dict:
                            if score > passage_score_dict[passage]:
                                passage_score_dict[passage] = score
                        else:
                            passage_score_dict[passage] = score
                    passage_score_dict = {
                        k: v
                        for k, v in sorted(
                            passage_score_dict.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    }
                    # separate into hits and scores
                    passage_hits = np.array(list(passage_score_dict.keys()))
                    passage_scores = np.array(list(passage_score_dict.values()))
                    # gap, elbow, or constant thresholding
                    if self.k == -1:
                        thresh_k = gap_thresholding(passage_scores)
                    elif self.k == -2:
                        thresh_k = elbow_thresholding(passage_scores)
                    elif self.k > 0:
                        thresh_k = constant_thresholding(passage_scores, self.k)
                    # get the valid passages
                    valid_passages = passage_hits[:thresh_k]
                    # create mask
                    mask = np.zeros(len(sorted_scores))
                    for i in range(len(sorted_scores)):
                        # if the sentence hit is in a valid passage, then it is valid
                        if (
                            corpus.sentence_passage_id[prior_hits[hits[i]], 0]
                            in valid_passages
                        ):
                            mask[i] = 1
                else:
                    # gap, elbow, or constant thresholding
                    if self.k == -1:
                        thresh_k = gap_thresholding(sorted_scores)
                    elif self.k == -2:
                        thresh_k = elbow_thresholding(sorted_scores)
                    elif self.k > 0:
                        thresh_k = constant_thresholding(sorted_scores, self.k)
                    # create mask
                    mask = np.zeros(len(sorted_scores))
                    mask[:thresh_k] = 1
                # apply mask
                scores = sorted_scores[mask == 1]
                hits = prior_hits[hits[mask == 1]]
            # otherwise, just take the top-k
            else:
                # if passage_search, then the top-k passages are retrieved
                if self.passage_search:
                    # adjust k for number of prior passages
                    adj_k = min(n_prior_passages, self.k)
                    # initialize unique passages seen and mask
                    unique_passages = []
                    mask = np.zeros(len(sorted_scores))
                    for i in range(len(sorted_scores)):
                        # get the current passage id
                        current_passage = corpus.sentence_passage_id[
                            prior_hits[hits[i]], 0
                        ]
                        if current_passage in unique_passages:
                            mask[i] = 1
                        else:
                            if len(unique_passages) < adj_k:
                                mask[i] = 1
                                unique_passages.append(current_passage)
                # otherwise, just take the top-k sentences
                else:
                    # adjust k for number of prior sentences
                    adj_k = min(n_prior_hits, self.k)
                    # create mask
                    mask = np.zeros(len(sorted_scores))
                    mask[:adj_k] = 1
                    # apply mask
                scores = sorted_scores[mask == 1]
                hits = prior_hits[hits[mask == 1]]
        else:
            scores, hits = prior_scores, prior_hits
        return scores, hits


class PoolRetriever:
    """
    a retriever that pools the results of multiple retrievers
    """

    def __init__(self, config):
        """
        initializer
        input: config (dict)
        """
        # set name and mode
        self.name = "pool"
        self.mode = "passage"
        # set config and build retrievers
        self.pooling = config["pooling"]
        self.k = config["k"]
        self.retriever_config = config["retriever_config"]
        self.build_retrievers()

    def build_retrievers(self):
        """
        builds retrievers from config
        """
        # initialize retrievers and weights
        self.retrievers = []
        self.weights = []
        self.retriever_histories = []
        # iterate through configs
        for config in self.retriever_config:
            # build retriever from config
            self.retrievers.append(build_retriever(config))
            # add weight
            self.weights.append(config["weight"])
            # add retrieval history
            self.retriever_histories.append(RetrievalHistory())
        # normalize weights
        total_weight = np.sum(self.weights)
        self.weights = [weight / total_weight for weight in self.weights]

    def retrieve(self, corpus, query, prior_scores, prior_hits, model_dict):
        """
        retrieves passages from corpus using a pool of retrievers
        input: corpus (Corpus)
               query (str)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               model_dict (dict[str:[CorpusModel, EmbeddingModel, CrossEncoderModel, None]])
        output: scores (np.ndarray)
                hits (np.ndarray)
        """
        # initialize scores and hits
        all_scores = []
        all_hits = []
        # iterate through retrievers
        for i, retriever in enumerate(self.retrievers):
            scores, hits = retriever.retrieve(
                corpus,
                query,
                prior_scores,
                prior_hits,
                model_dict[retriever.name],
            )
            self.retriever_histories[i].add(
                retriever.name,
                retriever.mode,
                scores,
                hits,
            )
            # convert scores and hits if necessary
            if retriever.mode == "sentence":
                self.retriever_histories[i].convert_sentences_to_passages(
                    corpus,
                    scores,
                    hits,
                    retriever.pooling,
                )
            scores, hits = (
                self.retriever_histories[i].history[-1].scores,
                self.retriever_histories[i].history[-1].hits,
            )
            # normalize scores
            scores = np.array(scores)
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = np.zeros(len(scores)) + 1e-6
            # append scores and hits
            all_scores.append(scores)
            all_hits.append(hits)
        # combine weighted scores
        hit_score_weight_dict = {}
        for i, (scores, hits, weight) in enumerate(
            zip(all_scores, all_hits, self.weights)
        ):
            for score, hit in zip(scores, hits):
                if hit not in hit_score_weight_dict:
                    hit_score_weight_dict[hit] = {"score": {}, "weight": {}}
                hit_score_weight_dict[hit]["score"][i] = score
                hit_score_weight_dict[hit]["weight"][i] = weight
        for hit in hit_score_weight_dict.keys():
            for i in range(len(self.retrievers)):
                if i not in hit_score_weight_dict[hit]["score"].keys():
                    hit_score_weight_dict[hit]["score"][i] = 1e-6
                    hit_score_weight_dict[hit]["weight"][i] = self.weights[i]
        weighted_hit_score_dict = {}
        for hit, score_weight_dict in hit_score_weight_dict.items():
            _scores = np.array(list(score_weight_dict["score"].values()))
            _weights = np.array(list(score_weight_dict["weight"].values()))
            if self.pooling == "max":
                weighted_hit_score_dict[hit] = np.max(_scores)
            elif self.pooling == "arithmetic_mean":
                weighted_hit_score_dict[hit] = np.average(_scores, weights=_weights)
            elif self.pooling == "geometric_mean":
                weighted_hit_score_dict[hit] = np.power(
                    np.prod(np.power(_scores, _weights)), 1 / np.sum(_weights)
                )
            elif self.pooling == "harmonic_mean":
                weighted_hit_score_dict[hit] = np.sum(_weights) / np.sum(
                    _weights / np.array(_scores)
                )
        # sort by weighted score
        weighted_hit_score_dict = {
            k: v
            for k, v in sorted(
                weighted_hit_score_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        # separate into scores and hits
        hits = np.array(list(weighted_hit_score_dict.keys()))
        scores = np.array(list(weighted_hit_score_dict.values()))
        # adjust k to be the minimum returned hits from a retriever
        adj_k = min(len(hits), self.k)
        # if k is negative, threshold
        if self.k < 1:
            # gap, elbow, or constant thresholding
            if self.k == -1:
                thresh_k = gap_thresholding(scores)
            elif self.k == -2:
                thresh_k = elbow_thresholding(scores)
            elif self.k > 0:
                thresh_k = constant_thresholding(scores, self.k)
            # create mask
            explicit_mask = np.zeros(len(scores))
            explicit_mask[:thresh_k] = 1
        else:
            # else the mask just goes up to k
            explicit_mask = np.zeros(len(scores))
            explicit_mask[:adj_k] = 1
        # apply mask
        scores = scores[explicit_mask == 1]
        hits = hits[explicit_mask == 1]
        return scores, hits


class ScoresAndHitsObject:
    """
    a container for storing scores and hits
    """

    def __init__(self):
        """
        empty initializer
        """
        pass

    def build(self, config, mode, scores, hits):
        """
        adds scores and hits
        input: config (str)
               scores (np.ndarray)
               hits (np.ndarray)
        """
        self.config = config
        self.mode = mode
        self.scores = scores
        self.hits = hits


class RetrievalHistory:
    """
    a constainer for storing retrieval history
    """

    def __init__(self):
        """
        initializer
        """
        self.history = []

    def add(self, config, mode, scores, hits):
        """
        adds scores and hits to history
        input: config (str)
               scores (np.ndarray)
               hits (np.ndarray)
        """
        scores_and_hits = ScoresAndHitsObject()
        scores_and_hits.build(config, mode, scores, hits)
        self.history.append(scores_and_hits)

    def convert_passages_to_sentences(self, corpus, prior_scores, prior_hits):
        """
        converts passage scores and hits to sentence scores and hits
        assigns passage score to all sentences in passage
        input: corpus (Corpus)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
        """
        sentence_bounds = corpus.passage_sentence_id[prior_hits]
        prior_sentence_scores = np.concatenate(
            [
                prior_score * np.ones(np.diff(start_end))
                for prior_score, start_end in zip(prior_scores, sentence_bounds)
            ]
        )
        prior_sentence_hits = np.concatenate(
            [np.arange(start, end) for start, end in sentence_bounds]
        )
        scores_and_hits = ScoresAndHitsObject()
        scores_and_hits.build(
            {"name": "passages_to_sentences"},
            "sentence",
            prior_sentence_scores,
            prior_sentence_hits,
        )
        self.history.append(scores_and_hits)

    def convert_sentences_to_passages(self, corpus, prior_scores, prior_hits, pooling):
        """
        converts sentence scores and hits to passage scores and hits
        assigns highest sentence score to passage
        input: corpus (Corpus)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               pooling (str) either "max" or "mean"
        """
        passage_ids = [
            (corpus.sentence_passage_id[i, 0], s)
            for i, s in zip(prior_hits, prior_scores)
        ]
        passage_dict = {}
        for i, s in passage_ids:
            if i in passage_dict.keys():
                passage_dict[i].append(s)
            else:
                passage_dict[i] = [s]
        if pooling == "max":
            passage_dict = {p: np.max(s) for p, s in passage_dict.items()}
        elif pooling == "arithmetic_mean":
            passage_dict = {p: np.mean(s) for p, s in passage_dict.items()}
        elif pooling == "geometric_mean":
            passage_dict = {
                p: np.prod(s) ** (1 / len(s)) for p, s in passage_dict.items()
            }
        elif pooling == "harmonic_mean":
            passage_dict = {
                p: len(s) / np.sum(1 / np.array(s)) for p, s in passage_dict.items()
            }
        passage_dict = {
            p: s
            for p, s in sorted(passage_dict.items(), key=lambda x: x[1], reverse=True)
        }
        prior_passage_scores = np.array(list(passage_dict.values()))
        prior_passage_hits = np.array(list(passage_dict.keys()))
        scores_and_hits = ScoresAndHitsObject()
        scores_and_hits.build(
            {"name": "sentences_to_passages"},
            "passage",
            prior_passage_scores,
            prior_passage_hits,
        )
        self.history.append(scores_and_hits)

    def get_result(self, corpus):
        """
        returns the final result of the full retrieval history
        provides final scores and hits alongside the full configuration chain
        input: corpus (Corpus)
        output: scores_and_hits (ScoresAndHitsObject)
        """
        scores_and_hits = ScoresAndHitsObject()
        if self.history[-1].mode == "passage":
            self.convert_passages_to_sentences(
                corpus,
                self.history[-1].scores,
                self.history[-1].hits,
            )
        scores, hits = self.history[-1].scores, self.history[-1].hits
        scores_and_hits.build(
            [h.config for h in self.history],
            [h.mode for h in self.history],
            scores,
            hits,
        )
        return scores_and_hits

    def reset(self):
        """
        resets the history
        """
        self.history = []


class PassageObject:
    """
    a container for storing passage information
    """

    def __init__(self):
        """
        empty initializer
        """
        pass

    def build_passage(
        self,
        passage,
        document_id,
        passage_id,
        document_passage_id,
        passage_sentence_id,
        score,
    ):
        """
        builds a passage
        input: passage (str)
               document_id (int)
               passage_id (int)
               document_passage_id (int)
               passage_sentence_id (list[int])
               score (list[float])
        """
        self.passage = passage
        self.document_id = document_id
        self.passage_id = passage_id
        self.document_passage_id = document_passage_id
        self.passage_sentence_id = passage_sentence_id
        self.score = score

    def set_verdict(self, verdict):
        """
        sets relevance verdict for the passage
        input: verdict (int)
        """
        self.verdict = verdict

    def get_serializable_passage(self):
        """
        returns a serializable version of the passage information
        """
        try:
            return {
                "passage": self.passage.text,
                "sentences": list([sentence.text for sentence in self.passage.sents]),
                "document_id": self.document_id,
                "passage_id": self.passage_id,
                "document_passage_id": self.document_passage_id,
                "passage_sentence_id": self.passage_sentence_id,
                "score": self.score,
            }
        except:
            print("Malformed passage, perhaps it was not built")


class PassagesObject:
    """
    a container for storing a list of passages
    """

    def __init__(self):
        """
        initializer
        """
        self.passages = []

    def build_passages(self, corpus, result, corpus_model):
        """
        builds passages from a result
        input: corpus (Corpus)
               result (ScoresAndHitsObject)
               corpus_model (CorpusModel)
        """
        # initialize unique passages seen
        unique_passage_ids = set([])
        # initialize passage dict
        passage_dict = {}
        # iterate through hits
        for i in range(len(result.hits)):
            # retrieve passage and sentence index for each hit
            j, k = corpus.sentence_passage_id[result.hits[i]]
            # add passage id to unique passage ids
            unique_passage_ids.add(j)
            # retrieve document id
            t = corpus.passage_document_id[j]
            # if the document id is in the passage dict
            if t in passage_dict.keys():
                # if the passage id is in the passage dict
                if j in passage_dict[t].keys():
                    # add the sentence index and score
                    passage_dict[t][j]["sentence_id"].append(k)
                    passage_dict[t][j]["score"].append(result.scores[i])
                else:
                    # otherwise, initialize the passage dict for the passage
                    passage_dict[t][j] = {
                        "passage": corpus.corpus[j],
                        "sentence_id": [k],
                        "score": [result.scores[i]],
                    }
            else:
                # otherwise, initialize the passage dict for the document
                passage_dict[t] = {
                    j: {
                        "sentence_id": [k],
                        "score": [result.scores[i]],
                    }
                }
        # generate unique passage docs
        passage_texts = {
            i: p
            for i, p in zip(
                unique_passage_ids,
                corpus.passage_generator(unique_passage_ids, corpus_model),
            )
        }
        # iterate through passage dict
        for t in passage_dict.keys():
            for j in passage_dict[t].keys():
                # initialize passage
                passage = PassageObject()
                # build passage
                passage.build_passage(
                    passage_texts[j],
                    t,
                    int(j),
                    int(j - corpus.document_passage_id[t][0]),
                    [int(x) for x in passage_dict[t][j]["sentence_id"]],
                    [float(x) for x in passage_dict[t][j]["score"]],
                )
                self.passages.append(passage)

    def sort_passages_by_score(self, pooling):
        """
        sorts the passages by score
        """
        if pooling == "max":
            self.passages = sorted(
                self.passages, key=lambda x: max(x.score), reverse=True
            )
        if pooling == "arithmetic_mean":
            self.passages = sorted(
                self.passages, key=lambda x: np.mean(x.score), reverse=True
            )
        if pooling == "geometric_mean":
            self.passages = sorted(
                self.passages,
                key=lambda x: np.prod(x.score) ** (1 / len(x.score)),
                reverse=True,
            )
        if pooling == "harmonic_mean":
            self.passages = sorted(
                self.passages,
                key=lambda x: len(x.score) / np.sum(1 / np.array(x.score)),
                reverse=True,
            )

    def sort_passages_by_temporal_position(self):
        """
        sorts the passages primarily by maximum document score and then secondarily by passage index
        """
        max_document_score = {}
        for passage in self.passages:
            if passage.document_id not in max_document_score:
                max_document_score[passage.document_id] = max(passage.score)
            else:
                max_document_score[passage.document_id] = max(
                    max_document_score[passage.document_id], max(passage.score)
                )
        self.passages = sorted(
            self.passages,
            key=lambda passage: (
                max_document_score[passage.document_id],
                -passage.passage_id,
            ),
            reverse=True,
        )

    def extend(self, passages):
        """
        extends the passages
        input: passages (PassagesObject)
        """
        self.passages.extend(passages.get_passages())

    def set_verdict(self, index, verdict):
        """
        sets the verdict for a passage by index
        input: index (int)
               verdict (int)
        """
        self.passages[index].set_verdict(verdict)

    def set_verdicts(self, verdicts):
        """
        sets the verdicts for all passages
        input: verdicts (list[int])
        """
        for i in range(len(verdicts)):
            self.passages[i].set_verdict(verdicts[i])

    def filter_by_verdict(self):
        """
        filters the passages by verdict
        """
        self.passages = [passage for passage in self.passages if passage.verdict == 1]

    def get_passages(self):
        """
        returns the passages
        """
        return self.passages

    def get_serializable_passages(self):
        """
        returns a serializable version of the passages
        """
        return [passage.get_serializable_passage() for passage in self.passages]

    def reset(self):
        """
        resets the passages
        """
        self.passages = []


class RetrieverChain:
    """
    a retriever chain that links retrievers together
    """

    def __init__(self, config):
        """
        initializer
        input: config (dict)
        """
        # set config, retrieval history, and build retriever chain
        self.config = config
        self.retrieval_history = RetrievalHistory()
        self.build_retriever_chain()

    def build_retriever_chain(self):
        """
        builds retriver chain from config
        """
        # initialize list
        self.retriever_chain = []
        if len(self.config) != 0:
            # iterate through config
            for config in self.config:
                self.retriever_chain.append(build_retriever(config))

    def update_config(self, config):
        """
        update config, reset retrieval history, and rebuild retriever chain
        """
        self.config = config
        self.retrieval_history.reset()
        self.build_retriever_chain()

    def reset_history(self):
        """
        resets the retrieval history
        """
        self.retrieval_history.reset()

    def retrieve(self, corpus, query, prior_scores, prior_hits, model_dict):
        """
        retrieves passages from corpus using a retriever chain
        input: corpus (Corpus)
               query (str)
               prior_scores (np.ndarray)
               prior_hits (np.ndarray)
               model_dict (dict[str:[CorpusModel, EmbeddingModel, dict[str:[CorpusModel, CrossEncodingModel]], None]])
        output: scores (np.ndarray)
        """
        print("Retrieving with Retriever Chain")
        # iterate through chain
        for i in range(len(self.retriever_chain)):
            # convert scores and hits if necessary
            if i > 0:
                if (
                    self.retriever_chain[i - 1].mode == "sentence"
                    and self.retriever_chain[i].mode == "passage"
                ):
                    self.retrieval_history.convert_sentences_to_passages(
                        corpus,
                        prior_scores,
                        prior_hits,
                        self.retriever_chain[i].pooling,
                    )
                    last = self.retrieval_history.get_last()
                    prior_scores, prior_hits = last.scores, last.hits
                elif (
                    self.retriever_chain[i - 1].mode == "passage"
                    and self.retriever_chain[i].mode == "sentence"
                ):
                    self.retrieval_history.convert_passages_to_sentences(
                        corpus, prior_scores, prior_hits
                    )
                    prior_scores, prior_hits = (
                        self.retrieval_history.history[-1].scores,
                        self.retrieval_history.history[-1].hits,
                    )
            # retrieve
            scores, hits = self.retriever_chain[i].retrieve(
                corpus,
                query,
                prior_scores,
                prior_hits,
                model_dict
                if self.retriever_chain[i].name == "pool"
                else model_dict[self.retriever_chain[i].name],
            )
            # add to history
            self.retrieval_history.add(
                self.config[i], self.retriever_chain[i].mode, scores, hits
            )
            # update prior scores and hits
            prior_scores, prior_hits = scores, hits
        # return final result
        result = self.retrieval_history.get_result(corpus)
        return result


class RetrievalHandler:
    """
    a retrieval handler that handles the retrieval process
    """

    def __init__(self, config):
        """
        initializer
        input: config (dict)
        """
        # set config and initialize retriever
        self.config = config
        self.initialize_retriever()

    def update_config(self, config):
        """
        update config and retriever
        input: config (dict)
        """
        self.config = config
        self.initialize_retriever()

    def initialize_retriever(self):
        """
        initialize retriever with config
        """
        self.retriever = RetrieverChain(self.config)

    def rank_sentences(
        self,
        corpus,
        query,
        corpus_model,
        embedding_model,
        cross_encoding_model,
    ):
        """
        rank sentences in corpus using retriever
        input: corpus (Corpus)
               query (str)
               corpus_model (CorpusModel)
               embedding_model (EmbeddingModel)
               cross_encoding_model (CrossEncoderModel)
        output: result (ScoresAndHitsObject)
        """
        # initialize model dict
        model_dict = {
            "bm25": corpus_model,
            "graph": None,
            "vector": embedding_model,
            "cross_encoder": {
                "corpus": corpus_model,
                "cross_encoder": cross_encoding_model,
            },
        }
        # retrieve
        total_docs = len(corpus.corpus_doc_index) - 1
        result = self.retriever.retrieve(
            corpus,
            query,
            np.ones(total_docs),
            np.arange(total_docs),
            model_dict,
        )
        # return result
        return result

    def validate_passages(
        self,
        passages,
        query,
        validation_model,
    ):
        """
        validates passage relevance using validation model
        input: query (str)
               passages (PassagesObject)
               validation_model (QueryModel)
        output: verdicts (list[int])
        """
        verdicts = []
        for passage in tqdm(passages.get_passages(), desc="Validating Passages"):
            verdict = validation_model.validate(passage.passage.text, query)
            try:
                verdict = int(verdict)
            except:
                verdict = -2
            verdicts.append(verdict)
        return verdicts

    def search(
        self,
        corpus,
        query,
        corpus_model,
        embedding_model,
        cross_encoding_model,
        validation_model,
    ):
        """
        searches the corpus for relevant passages to a query
        input: corpus (Corpus)
               query (str)
               corpus_model (CorpusModel)
               embedding_model (EmbeddingModel)
               cross_encoding_model (CrossEncoderModel)
               validation_model (QueryModel)
        output: passages (PassagesObject)
        """
        # clean query
        query = clean_text(query).replace("\n", " ").strip()
        # rank sentences in corpus
        result = self.rank_sentences(
            corpus,
            query,
            corpus_model,
            embedding_model,
            cross_encoding_model,
        )
        # intialize passages
        passages = PassagesObject()
        # build passages
        passages.build_passages(corpus, result, corpus_model)
        # if validation model is not none, validate passages
        if validation_model is not None:
            verdicts = self.validate_passages(
                passages,
                query,
                validation_model,
            )
            passages.set_verdicts(verdicts)
            passages.filter_by_verdict()
        # sort passages by score
        if "pooling" in self.config[-1]["parameters"].keys():
            passages.sort_passages_by_score(self.config[-1]["parameters"]["pooling"])
        else:
            passages.sort_passages_by_score("max")
        # reset retrieval history
        self.retriever.reset_history()
        return passages
