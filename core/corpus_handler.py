import os
import ujson as json
import gc
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from pypdf import PdfReader
from spacy.tokens import Doc, DocBin
from rank_bm25 import BM25Okapi, BM25Plus

##########################################################################
#### This module handles everything related to preprocessing a corpus ####
##########################################################################


class CorpusObject:
    """
    a container for all the data related to a corpus
    """

    def __init__(self):
        """
        empty initializer
        """
        pass

    def load_corpus_file(self, corpus_file, corpus_model):
        """
        loads a corpus from file into a list of byte strings
        input: corpus_file (str)
               corpus_model (CorpusModel)
        output: corpus (list[bytes])
        """
        doc_bin = DocBin(attrs=corpus_model.doc_bin_attrs).from_disk(corpus_file)
        return [
            passage.to_bytes() for passage in doc_bin.get_docs(corpus_model.model.vocab)
        ]

    def load_corpus_serial(self, corpus_path, corpus_model):
        """
        loads the corpus in series
        input: corpus_path (str)
               corpus_model (CorpusModel)
        output: corpus (np.array[uint8])
                doc_index (np.array[int64])
        """
        doc_bin_path = os.path.join(corpus_path, "corpus")
        corpus_files = [
            os.path.join(doc_bin_path, f) for f in sorted(os.listdir(doc_bin_path))
        ]
        corpus = bytearray()
        doc_index = [0]  # Start index of the first document
        for corpus_file in tqdm(
            corpus_files,
            total=len(corpus_files),
            desc="Loading Corpus",
            position=0,
            leave=True,
        ):
            for doc_bytes in self.load_corpus_file(corpus_file, corpus_model):
                corpus.extend(doc_bytes)
                doc_index.append(len(corpus))  # Start index of the next document
        return np.array(corpus, dtype=np.uint8), np.array(doc_index, dtype=np.int64)

    def load_corpus(
        self,
        corpus_path,
        corpus_model,
    ):
        """
        loads the core corpus data
        input: corpus_path (str)
               corpus_model (CorpusModel)
        """
        # path to list of relevant lemmas in vocabulary
        relevant_vocab_list_path = os.path.join(corpus_path, "relevant_vocab_list.pkl")
        # path to BM25 model
        passage_bm25_model_path = os.path.join(corpus_path, "passage_bm25_model.pkl")
        # path to document-passage id dictionary
        document_passage_id_path = os.path.join(corpus_path, "document_passage_id.pkl")
        # path to passage-document id dictionary
        passage_document_id_path = os.path.join(corpus_path, "passage_document_id.pkl")
        # path to embeddings for relevant lemmas in vocabulary
        relevant_vocab_embeddings_path = os.path.join(
            corpus_path, "relevant_vocab_embeddings.npy"
        )
        # path to passage-sentence id dictionary
        passage_sentence_id_path = os.path.join(corpus_path, "passage_sentence_id.npy")
        # path to sentence-passage id dictionary
        sentence_passage_id_path = os.path.join(corpus_path, "sentence_passage_id.npy")
        # load all other core corpus data
        with open(relevant_vocab_list_path, "rb") as f:
            self.relevant_vocab_list = pickle.load(f)
        with open(passage_bm25_model_path, "rb") as f:
            self.passage_bm25_model = pickle.load(f)
        with open(document_passage_id_path, "rb") as f:
            self.document_passage_id = pickle.load(f)
        with open(passage_document_id_path, "rb") as f:
            self.passage_document_id = pickle.load(f)
        self.relevant_vocab_embeddings = np.load(relevant_vocab_embeddings_path)
        self.passage_sentence_id = np.load(passage_sentence_id_path)
        self.sentence_passage_id = np.load(sentence_passage_id_path)
        # load corpus
        self.corpus, self.corpus_doc_index = self.load_corpus_serial(
            corpus_path,
            corpus_model,
        )

    def load_embedding(
        self,
        corpus_path,
        embedding_model_path,
    ):
        """
        loads the embedding data
        input: corpus_path (str)
        """
        name = embedding_model_path.split("/")[-1]
        print("Loading Embeddings")
        self.passage_embeddings = np.load(
            os.path.join(
                corpus_path,
                "{}_{}.npy".format("passage_embeddings", name),
            )
        )

    def load(
        self,
        corpus_path,
        embedding_model_path,
        corpus_model,
    ):
        """
        loads the full corpus data
        input: corpus_path (str)
        """
        self.load_corpus(corpus_path, corpus_model)
        self.load_embedding(corpus_path, embedding_model_path)

    def passage_generator(self, indices, corpus_model):
        """
        generator for yielding deserialized passages
        input: indices (list[int])
               corpus_model (CorpusModel)
        output: passage (spacy.tokens.doc.Doc)
        """
        for i in indices:
            start = self.corpus_doc_index[i]
            end = self.corpus_doc_index[i + 1]
            doc_bytes = self.corpus[start:end].tobytes()
            yield Doc(corpus_model.model.vocab).from_bytes(doc_bytes)


class CorpusHandler:
    """
    handles corpus creation and processing
    """

    def __init__(self):
        """
        empty initializer
        """
        pass

    def load_document(
        self,
        source_path,
        file_name,
    ):
        """
        loads a document
        supported formats:
        - .txt (each line is a passage)
        - .json (list of passages)
        - .pdf (each page is a passage)
        inputs: source_path (str)
                file_name (str)
        output: document (dict[str:str])
        """
        document_path = os.path.join(source_path, file_name)
        if document_path.lower().endswith(".txt"):
            with open(document_path, "r") as f:
                document = f.readlines()
        elif document_path.lower().endswith(".json"):
            with open(document_path, "r") as f:
                document = json.load(f)
        elif document_path.lower().endswith("pdf"):
            try:
                pdf_reader = PdfReader(document_path)
                num_passages = len(pdf_reader.pages)
                document = [
                    pdf_reader.pages[i].extract_text() for i in range(num_passages)
                ]
            except:
                print('The Document "{}" could not be loaded'.format(document_path))
        else:
            print('The Document "{}" could not be loaded'.format(document_path))
            document = []
        document = {"_id": file_name, "_doc": document}
        return document

    def load_documents(
        self,
        source_path,
    ):
        """
        loads documents in a source path
        input: source_path (str)
        output: clean_documents (list[dict[str:str]])
        """
        file_names = os.listdir(source_path)
        documents = [
            self.load_document(
                source_path,
                file_name,
            )
            for file_name in tqdm(
                file_names, desc="Loading Documents", position=0, leave=True
            )
        ]
        return documents

    def construct_corpus_text(self, documents):
        """
        constructs the corpus text from a list of documents
        input: documents (list[dict[str:str]])
        output: document_passage_id (dict[str:(int,int)])
                passage_document_id (list[str])
                corpus_text (list[str])
        """
        passage_document_id = []
        corpus_text = []
        # build corpus text and passage_document_id
        # the passage_document_id tracks which document each passage belongs to
        for document in documents:
            for passage in document["_doc"]:
                passage_document_id.append(document["_id"])
                corpus_text.append(passage)
        # build document_passage_id
        # the document_passage_id tracks the indices of the first and last passages of each document
        document_passage_id = {}
        for i in range(len(passage_document_id)):
            doc_id = passage_document_id[i]
            if doc_id in document_passage_id.keys():
                document_passage_id[doc_id].append(i)
            else:
                document_passage_id[doc_id] = [i]
        document_passage_id = {
            k: (min(v), max(v) + 1) for k, v in document_passage_id.items()
        }
        return document_passage_id, passage_document_id, corpus_text

    def construct_corpus(
        self,
        source_path,
        corpus_model,
    ):
        """
        process corpus with corpus model
        input: source_path (str)
               corpus_model (CorpusModel)
        output: document_passage_id (dict[str:(int,int)])
                passage_document_id (list[str])
                corpus (list[spacy.tokens.doc.Doc])
        """

        (
            document_passage_id,
            passage_document_id,
            corpus_text,
        ) = self.construct_corpus_text(self.load_documents(source_path))
        (
            sentence_passage_id,
            lemmas,
            unique_lemmas,
            doc_bin,
        ) = corpus_model.process_corpus(
            corpus_text,
        )
        return (
            document_passage_id,
            passage_document_id,
            sentence_passage_id,
            lemmas,
            unique_lemmas,
            doc_bin,
        )

    def extract_passage_sentence_ids(
        self,
        sentence_passage_id,
        total_passages,
    ):
        """
        extract sentence ids from corpus
        input: corpus (list[spacy.tokens.doc.Doc])
        """
        # initialize passage_sentence_id
        # given a passage index, passage_sentence_id returns the indices of the first and last sentences of that passage
        passage_sentence_id = [[] for _ in range(total_passages)]
        for i, (j, k) in enumerate(sentence_passage_id):
            passage_sentence_id[j].append(i)
        passage_sentence_id = np.array(
            [(min(v), max(v) + 1) for v in passage_sentence_id]
        )
        return passage_sentence_id

    def process_corpus(
        self,
        source_path,
        corpus_model,
    ):
        """
        process core corpus components
        input: source_path (str)
               corpus_model (CorpusModel)
        output: corpus_dict (dict[str:[...]])
        """
        # construct corpus
        (
            document_passage_id,
            passage_document_id,
            sentence_passage_id,
            lemmas,
            unique_lemmas,
            corpus_doc_bins,
        ) = self.construct_corpus(
            source_path,
            corpus_model,
        )
        # extract passage to sentence ids
        total_passages = len(lemmas)
        passage_sentence_id = self.extract_passage_sentence_ids(
            sentence_passage_id, total_passages
        )
        # construct bm25 model
        passage_bm25 = BM25Plus(lemmas)
        # process vocabulary
        (
            relevant_vocab_list,
            relevant_vocab_embeddings,
        ) = corpus_model.process_vocabulary(unique_lemmas)
        corpus_dict = {
            "corpus_doc_bins": corpus_doc_bins,
            "relevant_vocab_list": relevant_vocab_list,
            "relevant_vocab_embeddings": relevant_vocab_embeddings,
            "passage_bm25_model": passage_bm25,
            "document_passage_id": document_passage_id,
            "passage_document_id": passage_document_id,
            "passage_sentence_id": passage_sentence_id,
            "sentence_passage_id": sentence_passage_id,
        }
        return corpus_dict

    def embed_corpus(
        self,
        corpus,
        embedding_model,
        corpus_model,
    ):
        """
        encode corpus
        input: corpus (list[spacy.tokens.doc.Doc])
               embedding_model (EmbeddingModel)
        """
        print("Embedding Passages")
        embeddings = embedding_model.encode_passages(
            corpus,
            corpus_model,
        )
        return embeddings

    def save_corpus(
        self,
        corpus_path,
        corpus_dict,
    ):
        """
        save core corpus components
        input: corpus_path (str)
               corpus_dict (dict[str:[...]])
        """
        doc_bin_path = os.path.join(corpus_path, "corpus")
        if not os.path.exists(doc_bin_path):
            os.makedirs(doc_bin_path)
        padding = len(str(len(corpus_dict["corpus_doc_bins"])))
        for i, doc_bin in tqdm(
            enumerate(corpus_dict["corpus_doc_bins"]),
            desc="Saving Corpus",
            total=len(corpus_dict["corpus_doc_bins"]),
            position=0,
            leave=True,
        ):
            doc_bin.to_disk(
                os.path.join(
                    doc_bin_path, "corpus_{}.bin".format(str(i).zfill(padding))
                )
            )
        for d in [
            "relevant_vocab_list",
            "passage_bm25_model",
            "document_passage_id",
            "passage_document_id",
        ]:
            with open(os.path.join(corpus_path, "{}.pkl".format(d)), "wb") as f:
                pickle.dump(corpus_dict[d], f)
        for d in [
            "sentence_passage_id",
            "passage_sentence_id",
            "relevant_vocab_embeddings",
        ]:
            fname = os.path.join(corpus_path, "{}.npy".format(d))
            np.save(fname, corpus_dict[d])

    def save_embedding(self, corpus_path, embeddings, name):
        """
        save embeddings
        input: corpus_path (str)
               embedding_dict (dict[str:np.ndarray])
        """
        fname = os.path.join(
            corpus_path, "{}_{}.npy".format("passage_embeddings", name)
        )
        np.save(fname, embeddings)

    def make_corpus(
        self,
        source_path,
        corpus_path,
        corpus_model,
    ):
        """
        make core corpus components
        input: source path (str)
               corpus path (str)
               corpus model (CorpusModel)
        """
        corpus_dict = self.process_corpus(
            source_path,
            corpus_model,
        )
        if not os.path.exists(corpus_path):
            os.makedirs(corpus_path)
        self.save_corpus(
            corpus_path,
            corpus_dict,
        )

    def make_embedding(
        self,
        corpus_path,
        corpus,
        embedding_model,
        corpus_model,
    ):
        """
        make embeddings
        input: corpus_path (str)
               corpus (list[spacy.tokens.doc.Doc])
               embedding_model (EmbeddingModel)
        """
        embeddings = self.embed_corpus(corpus, embedding_model, corpus_model)
        if not os.path.exists(corpus_path):
            os.makedirs(corpus_path)
        self.save_embedding(
            corpus_path, embeddings, embedding_model.model_path.split("/")[-1]
        )

    def make(self, source_path, corpus_path, corpus_model, embedding_model):
        """
        make corpus and embeddings
        input: source_path (str)
               corpus_path (str)
               corpus_model (CorpusModel)
               embedding_model (EmbeddingModel)
        """
        self.make_corpus(
            source_path,
            corpus_path,
            corpus_model,
        )
        corpus = CorpusObject()
        corpus.load_corpus(corpus_path, corpus_model)
        self.make_embedding(
            corpus_path,
            corpus,
            embedding_model,
            corpus_model,
        )
