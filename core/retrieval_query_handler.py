from core.retrieval_handler import RetrievalHandler, PassagesObject
from core.query_handler import QueryHandler

########################################################################
#### This module combines the retrieval and query answering modules ####
########################################################################


class RetrievalQueryHandler:
    def __init__(self, retrieval_config):
        self.retrieval_handler = RetrievalHandler(retrieval_config)
        self.query_handler = QueryHandler()

    def update_retrieval_config(self, retrieval_config):
        self.retrieval_handler.update_config(retrieval_config)

    def search_answer(
        self,
        corpus,
        query,
        corpus_model,
        embedding_model,
        cross_encoding_model,
        validation_model,
        query_model,
    ):
        passages = self.retrieval_handler.search(
            corpus,
            query,
            corpus_model,
            embedding_model,
            cross_encoding_model,
            validation_model,
        )
        passages.sort_passages_by_temporal_position()
        answer = self.query_handler.answer(
            passages,
            query,
            query_model,
        )
        return passages, answer

    def multisearch_answer(
        self,
        corpus,
        query,
        retrieval_config,
        corpus_model,
        embedding_model,
        cross_encoding_model,
        validation_model,
        query_model,
    ):
        passages = PassagesObject()
        for c in corpus.keys():
            self.update_retrieval_config(retrieval_config[c])
            p = self.retrieval_handler.search(
                corpus[c],
                query,
                corpus_model,
                embedding_model,
                cross_encoding_model,
                validation_model,
            )
            passages.extend(p)
        passages.sort_passages_by_temporal_position()
        answer = self.query_handler.answer(
            passages,
            query,
            query_model,
        )
        return passages, answer
