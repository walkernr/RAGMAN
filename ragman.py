from core.corpus_handler import CorpusHandler, CorpusObject
from core.model_handler import ModelHandler
from core.retrieval_handler import RetrievalHandler
from core.retrieval_query_handler import RetrievalQueryHandler


class RAGMAN:
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
        ranking_batch_size=128,
        max_tokens=4096,
        max_new_tokens=512,
        retrieval_device="cpu",
        query_device="cuda:0",
    ):
        self.retrieval_config = retrieval_config
        self.embedding_model_path = embedding_model_path
        self.cross_encoding_model_path = cross_encoding_model_path
        self.query_model_path = query_model_path

        self.validate_retrieval = validate_retrieval

        self.n_proc = n_proc
        self.corpus_processing_batch_size = corpus_processing_batch_size
        self.corpus_encoding_batch_size = corpus_encoding_batch_size
        self.ranking_batch_size = ranking_batch_size
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.retrieval_device = retrieval_device
        self.query_device = query_device

        self.model_handler = ModelHandler()
        self.retrieval_query_handler = RetrievalQueryHandler(self.retrieval_config)

        self.load_models()

        self.corpus = {}

    def update_retrieval_config(self, retrieval_config):
        self.retrieval_config = retrieval_config
        self.retrieval_query_handler.update_retrieval_config(retrieval_config)

    def load_models(self):
        self.model_handler.load_corpus_model()
        self.model_handler.load_embedding_model(
            self.embedding_model_path, self.retrieval_device
        )
        self.model_handler.load_cross_encoding_model(
            self.cross_encoding_model_path, self.retrieval_device
        )
        if self.query_device is not None:
            self.model_handler.load_query_model(
                self.query_model_path,
                self.query_device,
            )

    def build_corpus(self, document_set):
        source_path = "./documents/{}".format(document_set)
        corpus_path = "./.corpus/{}".format(document_set)
        corpus_handler = CorpusHandler()
        corpus_model = self.model_handler.fetch_corpus_model(
            self.n_proc,
            self.corpus_processing_batch_size,
        )
        self.model_handler.unload_cross_encoding_model()
        if self.query_device is not None:
            self.model_handler.unload_query_model()
        if self.retrieval_device == "cpu":
            self.model_handler.unload_embedding_model()
            self.model_handler.load_embedding_model(self.embedding_model_path, "cuda:0")
        embedding_model = self.model_handler.fetch_embedding_model(
            self.corpus_encoding_batch_size,
        )
        corpus_handler.make(source_path, corpus_path, corpus_model, embedding_model)
        if self.retrieval_device == "cpu":
            self.model_handler.load_embedding_model(
                self.embedding_model_path, self.retrieval_device
            )
        self.model_handler.load_cross_encoding_model(
            self.cross_encoding_model_path, self.retrieval_device
        )
        if self.query_device is not None:
            self.model_handler.load_query_model(
                self.query_model_path,
                self.query_device,
            )

    def load_corpus(self, document_set):
        corpus_path = "./.corpus/{}".format(document_set)
        corpus_model = self.model_handler.fetch_corpus_model(
            self.n_proc,
            self.corpus_processing_batch_size,
        )
        self.corpus[document_set] = CorpusObject()
        self.corpus[document_set].load(
            corpus_path, self.embedding_model_path, corpus_model
        )

    def search(
        self,
        document_set,
        query,
    ):
        corpus_model = self.model_handler.fetch_corpus_model(
            self.n_proc,
            self.corpus_processing_batch_size,
        )
        embedding_model = self.model_handler.fetch_embedding_model(
            self.ranking_batch_size
        )
        cross_encoding_model = self.model_handler.fetch_cross_encoding_model(
            self.ranking_batch_size
        )
        if self.validate_retrieval and self.query_device is not None:
            validation_model = self.model_handler.fetch_query_model(
                self.max_tokens,
                self.max_new_tokens,
            )
        else:
            validation_model = None
        passages = self.retrieval_query_handler.retrieval_handler.search(
            self.corpus[document_set],
            query,
            corpus_model,
            embedding_model,
            cross_encoding_model,
            validation_model,
        )
        return passages

    def search_answer(
        self,
        document_set,
        query,
    ):
        corpus_model = self.model_handler.fetch_corpus_model(
            self.n_proc,
            self.corpus_processing_batch_size,
        )
        embedding_model = self.model_handler.fetch_embedding_model(
            self.ranking_batch_size
        )
        cross_encoding_model = self.model_handler.fetch_cross_encoding_model(
            self.ranking_batch_size
        )
        query_model = self.model_handler.fetch_query_model(
            self.max_tokens,
            self.max_new_tokens,
        )
        passages, answer = self.retrieval_query_handler.search_answer(
            self.corpus[document_set],
            query,
            corpus_model,
            embedding_model,
            cross_encoding_model,
            query_model if self.validate_retrieval else None,
            query_model,
        )
        return passages, answer
