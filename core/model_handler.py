import spacy
from spacy.tokens import Doc, DocBin
import lemminflect
import itertools
from scipy.interpolate import PchipInterpolator, splrep, splev
from cleantext import clean
import gc
import psutil
import os
import signal
from tqdm import tqdm
from FlagEmbedding import FlagModel, FlagReranker, LLMEmbedder
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import CrossEncoder
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
    pipeline,
)
import torch
import faiss
import numpy as np
from pympler import summary, muppy

######################################################################################
#### This module handles everything related to various models used in the system #####
######################################################################################


def get_memory_usage_summary():
    """
    returns a summary of the memory usage of the current process
    """
    objects = muppy.get_objects()
    sum_ = summary.summarize(objects)
    return sum_


def get_memory_usage_diff(sum_before, sum_after):
    """
    returns the difference between two memory usage summaries
    input: sum_before (summary)
           sum_after (summary)
    output: diff (summary)
    """
    diff = summary.get_diff(sum_before, sum_after)
    return diff


def clean_text(text):
    """
    provides a clean version of input text
    input: text (str)
    output: clean text (str)
    """
    clean_text = text.strip()
    clean_text = clean(
        clean_text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=False,
        no_emails=False,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        lang="en",
    )
    return clean_text


def clean_texts(texts):
    """
    provides a clean version of input texts
    input: texts (list[str])
    output: clean_texts (list[str])
    """
    clean_texts = [clean_text(text) for text in texts]
    return clean_texts


def sigmoid(x):
    """
    returns the sigmoid of input x (1/(1+e^-x) ))
    input: x (np.ndarray)
    output: fx (np.ndarray)
    """
    fx = 1 / (1 + np.exp(-x))
    return fx


def constant_thresholding(similarities, threshold):
    """
    provides the cutoff index for a given constant threshold of similarities
    input: similarities (np.ndarray)
           threshold (float)
    output: index (int)
    """
    if len(similarities) > 5:
        index = np.searchsorted(-similarities, -threshold, side="left")
    else:
        index = len(similarities)
    return index


def generate_smooth_gradient(domain, similarities, order):
    """
    provides a smooth gradient of the similarities of a given order using a cubic spline
    input: domain (np.ndarray)
           similarities (np.ndarray)
           order (int)
    output: smooth gradient (np.ndarray)
    """
    f = splrep(
        domain, similarities, k=3, s=len(similarities) - np.sqrt(2 * len(similarities))
    )
    if order == 0:
        return splev(domain, f)
    else:
        return splev(domain, f, der=order)


def generate_smooth_gradient_monotonic(domain, similarities, order):
    """
    provides a smooth gradient of the similarities of a given order using a monotonic pchip interpolator
    WARNING: only the first gradient is guaranteed to be continuous
    input: domain (np.ndarray)
           similarities (np.ndarray)
           order (int)
    output: smooth gradient (np.ndarray)
    """
    f = PchipInterpolator(domain, similarities)
    if order == 0:
        return f(domain)
    else:
        return f.derivative(order)(domain)


def gap_thresholding(similarities):
    """
    provides the thresholding index via finding the highest velocity dropoff in similarity
    input: similarities (np.ndarray)
    output: index (int)
    """
    if len(similarities) >= 5:
        domain = np.arange(len(similarities))
        first_derivative = generate_smooth_gradient_monotonic(domain, similarities, 1)
        index = np.argmin(first_derivative) + 1
    else:
        index = len(similarities)
    return index


def elbow_thresholding(similarities):
    """
    provides the thresholding index via finding the largest decrease of acceleration in similarity
    input: similarities (np.ndarray)
    output: index (int)
    """
    if len(similarities) >= 5:
        domain = np.arange(len(similarities))
        second_derivative = generate_smooth_gradient_monotonic(domain, similarities, 2)
        index = np.argmax(second_derivative) + 1
    else:
        index = len(similarities)
    return index


def bin_packing_indices(lengths, threshold, overlap):
    """
    gets the optimal bin packing indices for a list of lengths
    input: lengths (list[int])
           threshold (int)
           overlap (int)
    output: bins (list[tuple[int, int]])
    """
    bins = []
    start = 0
    current_sum = 0
    for end, length in enumerate(lengths):
        current_sum += length
        if current_sum > threshold:
            bins.append((start, end))
            overlap_start = end
            overlap_sum = lengths[overlap_start]
            while (
                overlap_start > start
                and overlap_sum + lengths[overlap_start - 1] <= overlap
            ):
                overlap_start -= 1
                overlap_sum += lengths[overlap_start]
            start = overlap_start
            current_sum = sum(lengths[start : end + 1])
    bins.append((start, len(lengths)))
    return bins


def format_sequence(sequence):
    """
    formats a sequence for based on trailing whitespace
    input: sequence (str)
    output: modified_sequence (str)
    """
    if sequence[-2:] == "\n" or sequence[-1:] == " ":
        modified_sequence = sequence
    else:
        modified_sequence = "{} ".format(sequence)
    return modified_sequence


def segment_text(texts, tokenizer, token_limit):
    """
    segments the inputs texts into bins according to a token limit
    input: texts (list[str])
           tokenizer (AutoTokenizer)
           token_limit (int)
    output: segments (list[str])
    """
    lengths = [
        tokenizer(text, padding=False, truncation=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        for text in texts
    ]
    bins = bin_packing_indices(lengths, token_limit, token_limit // 4)
    uncombined_segments = [texts[indices[0] : indices[1] + 1] for indices in bins]
    segments = [
        "".join([format_sequence(sequence) for sequence in segment])
        for segment in uncombined_segments
    ]
    return segments


class CPU:
    """
    context handler for disabling GPU
    """

    def __enter__(self):
        """
        on entering context, disable CUDA availability
        """
        # save the initial state of CUDA availability and then set it to false on entrance
        self.original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        return self

    def __exit__(self, *args):
        """
        on existing context, restore CUDA availability
        """
        # restore the original CUDA availability on exit
        torch.cuda.is_available = self.original_is_available


class CorpusModel:
    """
    handles the usage of the spaCy model for processing a corpus
    """

    def __init__(self):
        """
        initializes the corpus model
        """
        # these pipes are not needed and are disabled for processing
        # unfortunately, tok2vec is still needed to get decent sentence segmentation
        self.disabled_pipes = [
            "ner",
            "entity_linker",
            "entity_ruler",
            "textcat",
            "textcat_multilabel",
            "transformer",
            "trainable_lemmatizer",
            "sentencizer",
            "transformer",
        ]
        # there are additional pipes that are unneeded for processing the vocabulary
        self.disabled_pipes_vocabulary = [
            *self.disabled_pipes,
            "tagger",
            "parser",
            "lemmatizer",
            "morphologizer",
            "attribute_ruler",
            "senter",
        ]
        # dic bin attributes make sure that we can retain the ability to recover text and sentences
        # this is all that's needed after pre-processing
        self.doc_bin_attrs = ["ORTH", "SENT_START"]
        # load the model
        self.load_model()

    def load_model(self):
        """
        loads the model
        """
        # load the model (we use the large version to have more vectors)
        self.model = spacy.load("en_core_web_lg", disable=self.disabled_pipes)

    def set_n_proc(self, n_proc):
        """
        sets the number of processes for the pipe
        input: n_proc (int)
        """
        self.n_proc = n_proc

    def set_batch_size(self, batch_size):
        """
        sets the batch size for the pipe
        input: batch_size (int)
        """
        self.batch_size = batch_size

    def batch_texts_generator(self, texts):
        """
        generator for batching the corpus
        input: text (list[str])
        output: batch (list[str])
        """
        # generator for batching the corpus
        for i in range(0, len(texts), self.batch_size):
            yield texts[i : i + self.batch_size]

    def shutdown_child_processes(self):
        """
        shuts down all child processes of the current process
        """
        # experimental since sometimes it seems spaCy fails to shut down its child processes
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGTERM)

    def reload(self):
        """
        reloads the model to free up memory
        """
        # necessary since the voabulary strings grow with each batch, which gets excessive with a large corpus
        # self.shutdown_child_processes()
        # delete model and garbage collect
        del self.model
        gc.collect()
        # reload model
        self.load_model()

    def extract_lemmas(self, doc):
        """
        extracts a list of lemmas from a spaCy doc
        input: doc (spacy.tokens.Doc)
        output: lemmas (list[str])
        """
        return [
            t._.lemma().strip().lower()
            for t in doc
            if (not t.is_stop and not t.is_punct and not t.is_space and t.text != "")
        ]

    def process_corpus(self, texts):
        """
        processes a corpus using spaCy into a docbin
        input: texts (list[str])
        output: sentence_passage_id (np.ndarray)
                lemmas (list[list[str]])
                unique_lemmas (list[str])
                doc_bins (list[spacy.tokens.DocBin])
        """
        # initialize corpus
        doc_bins = []
        lemmas = []
        sentence_passage_id = []
        unique_lemmas = set()
        # get total number of batches
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        # iterate through batches
        i = 0
        for batch in tqdm(
            self.batch_texts_generator(texts),
            total=total_batches,
            desc="Processing Passages in Batches",
            position=0,
            leave=True,
        ):
            doc_bin = DocBin(attrs=self.doc_bin_attrs)
            # iterate through docs in batch
            for doc in tqdm(
                self.model.pipe(clean_texts(batch), n_process=self.n_proc),
                desc="Processing Batch",
                total=len(batch),
                position=1,
                leave=False,
            ):
                if doc.text == "":
                    doc = self.model("INVALID PASSAGE")
                sentence_passage_id.extend(
                    [(i, j) for j in range(len(list(doc.sents)))]
                )
                ls = self.extract_lemmas(doc)
                lemmas.append(ls)
                unique_lemmas.update(set(ls))
                doc_bin.add(doc)
                i += 1
            # reload model to free memory
            doc_bins.append(doc_bin)
            self.reload()
        sentence_passage_id = np.array(sentence_passage_id)
        unique_lemmas = list(unique_lemmas)
        return sentence_passage_id, lemmas, unique_lemmas, doc_bins

    def process_vocabulary(self, unique_lemmas):
        """
        processes the vocabulary of a corpus by getting the vectors for all relevant lemmas
        input: unique_lemmas (list[str])
        output: relevant_vocab_list (list[str])
                relevant_vocab_embeddings (np.ndarray)
        """
        # initialize relevant vocab
        relevant_vocab_list = []
        relevant_vocab_embeddings = []
        self.set_batch_size(10 * self.batch_size)
        total_batches = (len(unique_lemmas) + self.batch_size - 1) // (self.batch_size)
        # add vectors for all relevant lemmas to relevant vocab if they have a vector and are a single token
        for batch in tqdm(
            self.batch_texts_generator(unique_lemmas),
            total=total_batches,
            desc="Processing Vocabulary in Batches",
            position=0,
            leave=True,
        ):
            # iterate through vocabulary in batch
            for doc in tqdm(
                self.model.pipe(
                    clean_texts(batch),
                    disable=self.disabled_pipes_vocabulary,
                    n_process=self.n_proc,
                ),
                desc="Processing Batch",
                total=len(batch),
                position=1,
                leave=False,
            ):
                if len(doc) == 1:
                    token = doc[0]
                    if token.has_vector:
                        relevant_vocab_list.append(token.text)
                        relevant_vocab_embeddings.append(token.vector)
            # reload model to free memory
            self.reload()
        relevant_vocab_embeddings = np.array(
            relevant_vocab_embeddings, dtype=np.float32
        )
        # normalize vectors
        faiss.normalize_L2(relevant_vocab_embeddings)
        self.set_batch_size(self.batch_size // 10)
        return relevant_vocab_list, relevant_vocab_embeddings

    def extract_keywords(
        self,
        query,
    ):
        """
        extracts keywords from a query
        input: query (str)
        output: keywords (list[str])
        """
        query_tokens = self.model(clean_text(query).replace("\n", " ").strip())
        keywords = self.extract_lemmas(query_tokens)
        keyword_frequency_dict = {}
        for keyword in keywords:
            if keyword in keyword_frequency_dict:
                keyword_frequency_dict[keyword] += 1
            else:
                keyword_frequency_dict[keyword] = 1
        return keyword_frequency_dict

    def get_similar_keywords(
        self,
        corpus,
        keyword_frequency_dict,
        keyword_k,
    ):
        """
        retrieves similar keywords to a list of keywords
        input: corpus (CorpusObject)
               keywords (list[str])
        output: expanded_keywords (list[str])
        """
        n_relevant_vocab = len(corpus.relevant_vocab_list)
        keyword_set = list(keyword_frequency_dict.keys())
        expanded_keyword_dict = {kw: [kw] for kw in keyword_set}
        if keyword_k != 0:
            kw = []
            kv = []
            for doc in self.model.pipe(keyword_set, n_process=self.n_proc):
                if len(doc) == 1:
                    token = doc[0]
                    if token.has_vector:
                        kw.append(token.text)
                        kv.append(token.vector)
            if len(kv) > 0:
                kv = np.array(kv, dtype=np.float32)
                faiss.normalize_L2(kv)
                index = faiss.IndexFlatIP(corpus.relevant_vocab_embeddings.shape[1])
                index.add(corpus.relevant_vocab_embeddings)
                scores, hits = index.search(kv, n_relevant_vocab)
                if keyword_k < 1 and keyword_k > -3:
                    for i in range(len(kv)):
                        if corpus.relevant_vocab_list[hits[i, 0]] == kw[i]:
                            s = scores[i, 1:]
                            h = hits[i, 1:]
                        else:
                            s = scores[i]
                            h = hits[i]
                        if keyword_k == -1:
                            thresh_k = gap_thresholding(s)
                        elif keyword_k == -2:
                            thresh_k = elbow_thresholding(s)
                        else:
                            thresh_k = constant_thresholding(s, keyword_k)
                        expanded_keyword_dict[kw[i]].extend(
                            [corpus.relevant_vocab_list[h[j]] for j in range(thresh_k)]
                        )
                elif keyword_k == -3:
                    for i in range(len(kv)):
                        if kw[i] not in corpus.relevant_vocab_list:
                            expanded_keyword_dict[kw[i]].append(
                                corpus.relevant_vocab_list[hits[i, 0]]
                            )
                else:
                    for i in range(len(kv)):
                        sk = 0
                        j = 0
                        while sk < keyword_k and j < n_relevant_vocab:
                            if corpus.relevant_vocab_list[hits[i, j]] != kw[i]:
                                expanded_keyword_dict[kw[i]].append(
                                    corpus.relevant_vocab_list[hits[i, j]]
                                )
                                sk += 1
                            j += 1
            else:
                print("No suitable keyword vectors found")
        expanded_keywords = []
        for kw in keyword_set:
            for ekw in expanded_keyword_dict[kw]:
                expanded_keywords.extend(keyword_frequency_dict[kw] * [ekw])
        expanded_keywords = sorted(list(set(expanded_keywords)))
        return expanded_keywords


class EmbeddingModel:
    """
    handles the usage of embedding models to encode a corpus/query
    """

    def __init__(self, model_path, device):
        """
        initializes the embedding model
        input: model_path (str)
               device (str)
        """
        self.model_path = model_path
        self.device = device
        self.load_model()

    def set_batch_size(self, batch_size):
        """
        sets the batch size for inference
        input: batch_size (int)
        """
        self.batch_size = batch_size

    def load_model(self):
        """
        loads the model
        currently supported models:
        - BAAI/bge-large-en-v1.5
        - BAAI/llm-embedder
        - hkunlp/instructor-xl
        """
        # set instructions/task as necessary
        if self.model_path.startswith("BAAI/bge"):
            instruction = "Represent this sentence for searching relevant passages: "
        elif self.model_path == "BAAI/llm-embedder":
            self.task = "qa"
        elif self.model_path.startswith("hkunlp/instructor"):
            self.query_instruction = (
                "Represent the question for retrieving supporting documents: "
            )
            self.passage_instruction = "Represent the document for retrieval: "
        else:
            print("Invalid model selection, defaulting to BAAI/bge-large-en-v1.5")
            self.model_path = "BAAI/bge-large-en-v1.5"
            instruction = "Represent this sentence for searching relevant passages: "
        if "cuda" not in self.device:
            with CPU():
                # load with CPU if device is not CUDA (needed since these models automatically load to CUDA if available)
                # fp16 is false if using the model on CPU
                if self.model_path.startswith("BAAI/bge"):
                    self.model = FlagModel(
                        model_name_or_path=self.model_path,
                        pooling_method="cls",
                        normalize_embeddings=True,
                        query_instruction_for_retrieval=instruction,
                        use_fp16=False,
                    )
                elif self.model_path == "BAAI/llm-embedder":
                    self.model = LLMEmbedder(
                        self.model_path,
                        use_fp16=False,
                    )
                elif self.model_path.startswith("hkunlp/instructor"):
                    self.model = INSTRUCTOR(self.model_path, device=self.device)
        else:
            if self.model_path.startswith("BAAI/bge"):
                self.model = FlagModel(
                    model_name_or_path=self.model_path,
                    pooling_method="cls",
                    normalize_embeddings=True,
                    query_instruction_for_retrieval=instruction,
                    use_fp16=True,
                )
            elif self.model_path == "BAAI/llm-embedder":
                self.model = LLMEmbedder(
                    self.model_path,
                    use_fp16=True,
                )
            elif self.model_path.startswith("hkunlp/instructor"):
                self.model = INSTRUCTOR(self.model_path, device=self.device)

    def encode_query(self, query):
        """
        encodes a query using the embedding model
        input: query (str)
        output: encoded_query (np.ndarray)
        """
        # clean the query
        clean_query = [clean_text(query).replace("\n", " ").strip()]
        # encoding command changes depending on the model
        if self.model_path.startswith("BAAI/bge"):
            encoded_query = self.model.encode_queries(
                clean_query, batch_size=self.batch_size
            )
        elif self.model_path == "BAAI/llm-embedder":
            encoded_query = self.model.encode_queries(
                clean_query, task=self.task, batch_size=self.batch_size
            )
        elif self.model_path.startswith("hkunlp/instructor"):
            encoded_query = self.model.encode(
                [[self.query_instruction, clean_query]],
            )
        # convert to float32 and reshape for normalization
        encoded_query = encoded_query.astype("float32").reshape(1, -1)
        # normalize the embedding
        faiss.normalize_L2(encoded_query)
        return encoded_query

    def clean_passage_batch_generator(self, corpus, corpus_model):
        """
        generator for yielding clean plain text passages
        input: passages (generator[Doc])
        output: clean_passage (generator[str])
        """
        total_docs = len(corpus.corpus_doc_index) - 1
        for start_idx in range(0, total_docs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_docs)
            indices = list(range(start_idx, end_idx))
            batch = [
                passage.text.replace("\n", " ").strip()
                for passage in corpus.passage_generator(indices, corpus_model)
            ]
            yield batch

    def encode_batch(self, passages):
        if self.model_path.startswith("BAAI/bge"):
            encoded_passages = self.model.encode(passages, batch_size=self.batch_size)
        elif self.model_path == "BAAI/llm-embedder":
            encoded_passages = self.model.encode_keys(
                passages, task=self.task, batch_size=self.batch_size
            )
        elif self.model_path.startswith("hkunlp/instructor"):
            passages = [[self.passage_instruction, passage] for passage in passages]
            encoded_passages = self.model.encode(
                passages,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
        return encoded_passages

    def encode_passages(self, corpus, corpus_model):
        """
        encodes passages using the embedding model
        input: corpus (CorpusObject)
               corpus_model (CorpusModel)
        output: encoded_passages (np.ndarray)
        """
        # encode the passages
        total = np.ceil((len(corpus.corpus_doc_index) - 1) / self.batch_size).astype(
            int
        )
        encoded_passages = np.vstack(
            [
                self.encode_batch(batch)
                for batch in tqdm(
                    self.clean_passage_batch_generator(corpus, corpus_model),
                    total=total,
                    desc="Encoding Passages",
                    position=0,
                    leave=True,
                )
            ]
        )
        # convert to float32 for normalization
        encoded_passages = encoded_passages.astype("float32")
        # normalize the embeddings
        faiss.normalize_L2(encoded_passages)
        return encoded_passages


class CrossEncodingModel:
    """
    handles the usage of a cross-encoding model to score (query, sentence) pairs
    """

    def __init__(self, model_path, device):
        """
        initializes the cross-encoding model
        input: model_path (str)
               device (str)
        """
        self.model_path = model_path
        self.device = device
        self.load_model()

    def set_batch_size(self, batch_size):
        """
        sets the batch size for inference
        input: batch_size (int)
        """
        self.batch_size = batch_size

    def load_model(self):
        """
        loads the model
        currently supported models:
        - BAAI/bge-reranker-large
        - cross-encoder/ms-marco-MiniLM-L-12-v2
        """
        if self.model_path not in [
            "BAAI/bge-reranker-large",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ]:
            print("Invalid model selection, defaulting to BAAI/bge-reranker-large")
            self.model_path = "BAAI/bge-reranker-large"
        if "cuda" not in self.device:
            with CPU():
                # load with cpu if device is not CUDA (needed since these models automatically load to CUDA if available)
                # fp16 is false if using the model on CPU
                if self.model_path == "BAAI/bge-reranker-large":
                    self.model = FlagReranker(
                        self.model_path,
                        use_fp16=False,
                    )
                elif self.model_path == "cross-encoder/ms-marco-MiniLM-L-12-v2":
                    self.model = CrossEncoder(
                        self.model_path,
                        max_length=512,
                    )
        else:
            if self.model_path == "BAAI/bge-reranker-large":
                self.model = FlagReranker(
                    self.model_path,
                    use_fp16=True,
                )
            elif self.model_path == "cross-encoder/ms-marco-MiniLM-L-12-v2":
                self.model = CrossEncoder(
                    self.model_path,
                    max_length=512,
                )

    def cross_encode_query_with_sentences(
        self, corpus, query, prior_hits, corpus_model
    ):
        """
        calculates scores for (query, sentence) pairs using the cross-encoding model
        input: corpus (CorpusObject)
               query (str)
               prior_hits as sentence indices (list[int])
               corpus_model (CorpusModel)
        output: scores (np.ndarray)
        """
        # clean the query
        clean_query = clean_text(query).replace("\n", " ").strip()
        # identify unique passages
        unique_passage_ids = list(
            set([corpus.sentence_passage_id[i][0] for i in prior_hits])
        )
        # build passage cache
        passage_cache = {
            i: p
            for i, p in zip(
                unique_passage_ids,
                corpus.passage_generator(unique_passage_ids, corpus_model),
            )
        }
        # initialize cleaned sentences
        clean_sentences = []
        # get cleaned sentences using cached passages
        for i in prior_hits:
            u, v = corpus.sentence_passage_id[i]
            sentence = list(passage_cache[u].sents)[v]
            clean_sentences.append(sentence.text.replace("\n", " ").strip())
        # initialize inputs
        inputs = [(clean_query, clean_sentence) for clean_sentence in clean_sentences]
        # compute logits
        if self.model_path == "BAAI/bge-reranker-large":
            logits = self.model.compute_score(
                inputs,
                batch_size=self.batch_size,
            )
        elif self.model_path == "cross-encoder/ms-marco-MiniLM-L-12-v2":
            logits = self.model.predict(
                inputs,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
        # apply sigmoid transformation to logits to get scores
        scores = sigmoid(np.array(logits))
        return scores


class QueryModel:
    """
    handles the usage of an LLM for answer synthesis of a query conditioned on retrieved contexts
    """

    def __init__(self, model_path, model_file, max_tokens, device):
        """
        initializes the LLM
        input: model_path (str)
               device (str)
        """
        self.model_path = model_path
        self.model_file = model_file
        self.device = device
        self.max_tokens = max_tokens
        self.load_model()
        self.set_prompts()

    def load_model(self):
        """
        loads the model
        currently supported models (quantized):
        - TheBloke/Llama-2-7b-Chat-GPTQ
        - TheBloke/Mistral-7B-OpenOrca-GPTQ
        - TheBloke/neural-chat-7B-v3-3-GPTQ
        - TheBloke/Llama-2-7b-Chat-GGUF
        - TheBloke/Mistral-7B-OpenOrca-GGUF
        - TheBloke/neural-chat-7B-v3-3-GGUF
        """
        if self.model_path not in [
            "TheBloke/Llama-2-7b-Chat-GPTQ",
            "TheBloke/Mistral-7B-OpenOrca-GPTQ",
            "TheBloke/neural-chat-7B-v3-3-GPTQ",
            "TheBloke/Llama-2-7b-Chat-GGUF",
            "TheBloke/Mistral-7B-OpenOrca-GGUF",
            "TheBloke/neural-chat-7B-v3-3-GGUF",
        ]:
            print(
                "Invalid model selection, defaulting to TheBloke/Llama-2-7b-Chat-GPTQ"
            )
            self.model_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
        # load tokenizer
        if self.model_file is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
            )
            self.ctransformers = False
        else:
            from ctransformers import (
                AutoTokenizer as CAutoTokenizer,
                AutoModelForCausalLM as CAutoModelForCausalLM,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path.replace(
                    "GGUF",
                    "GPTQ",
                ).replace(
                    "GGML",
                    "GPTQ",
                )
            )
            self.ctransformers = True
        self.gptq = "GPTQ" in self.model_path
        self.gguf_ggml = "GGUF" in self.model_path or "GGML" in self.model_path
        # load model
        if self.ctransformers:
            self.model = CAutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_file=self.model_file,
                context_length=self.max_tokens,
                hf=True,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )
            if self.gptq:
                from auto_gptq import exllama_set_max_input_length

                self.model = exllama_set_max_input_length(self.model, self.max_tokens)
        # set pad token id to eos token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # set model to evaluation mode
        self.model.eval()

    def set_prompts(self):
        """
        initializes the prompts used with the LLM
        """
        # set chat instruction template
        if "TheBloke/Llama-2-7b-Chat" in self.model_path:
            instruction_template = "[INST] <<SYS>>\n{}\n<</SYS>>\n{}[/INST]"
        if "TheBloke/Mistral-7B-OpenOrca" in self.model_path:
            instruction_template = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        if "TheBloke/neural-chat-7B-v3-3" in self.model_path:
            instruction_template = (
                "### System:\n{}\n\n\n### User:\n{}\n\n\n### Assistant:\n"
            )
        # system instruction message for query answering
        system_query_instruction = (
            "You are a highly intelligence and accurate context-based question-answering assistant. "
            "You provide accurate and helpful responses to user queries based on the context provided by a context retrieval subsystem. "
            "Focus on directly answering the question instead of providing conversational responses. ",
            "Strive to understand and respond to every question, even if it is not perfectly formed or clear. ",
            "If a question is truly unanswerable due to factual inaccuracies or incoherence, gently clarify these issues with the user. ",
            "If a question is beyond your knowledge base, be honest about your limitations, but refrain from sharing false or misleading information.",
        )
        # system instruction message for answer aggregation
        system_aggregation_instruction = (
            "You are a highly intelligent and accurate answer-aggregation assistant. "
            "You will be provided with a question and an enumerated list of answers that were generated from contexts provided by a context retrieval subsystem. "
            "Consolidate the answers into a single comprehensive and consistent answer that accurately addresses the question. "
            "Do not cite the original answers, only provide the consolidated answer. "
            "Focus on directly answering the question instead of providing conversational responses. ",
            "Strive to understand and respond to every question, even if it is not perfectly formed or clear. ",
            "If a question is truly unanswerable due to factual inaccuracies or incoherence, gently clarify these issues with the user. ",
            "If a question is beyond your knowledge base, be honest about your limitations, but refrain from sharing false or misleading information.",
        )
        # system instruction message for context validation
        system_validation_instruction = (
            "You are a highly intelligent and accurate context-validation assistant. "
            "You will be provided with a question and a passage that was generated from a context retrieval subsystem. "
            "Determine whether the passage is relevant to the question. "
            "Focus on directly answering the question instead of providing conversational responses. ",
            "Strive to understand and respond to every question/context pair, even if they are not perfectly formed or clear. ",
            'If the provided context is relevant to the question, respond with "1". '
            'If the provided context is not relevant to the question, respond with "-1". ',
            'If it is unclear if the provided context is relevant to the question, respond with "0". ',
            "You must not respond with any other values or anything other than those values. ",
        )
        # construct prompt templates from chat templates and system instructions
        self.query_prompt = instruction_template.format(
            system_query_instruction,
            "{}\n\nBased on the information above, answer the question: {}",
        )
        self.aggregation_prompt = instruction_template.format(
            system_aggregation_instruction,
            "{}\n\n{}\n\nPlease synthesize and consolidate the answers from the different sources into a single, comprehensive, and consistent answer that accurately addresses the question.",
        )
        self.validation_prompt = instruction_template.format(
            system_validation_instruction,
            '{}\n\nBased on the information above can we conclude that the question "{}" is relevant?',
        )

    def set_config(self, max_new_tokens):
        """
        initializes generation config for the LLM
        input: max_new_tokens (int)
        """
        # maximum new tokens (to be generated)
        self.max_new_tokens = max_new_tokens
        # set max input length if using Mistral
        if not self.ctransformers:
            # initialize generation config
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
            # deterministic generation
            self.generation_config.do_sample = False
            self.generation_config.num_beams = 1
            # token limits
            self.generation_config.max_new_tokens = max_new_tokens
            self.generation_config.min_new_tokens = 0
        else:
            self.generation_config = {
                "do_sample": 0,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 0,
            }

    def calculate_max_context_length(
        self,
        prompt,
    ):
        """
        calculates the maximum context length based on the prompted query
        input: prompt (str)
        output: max_context_length (int)
        """
        prompt_length = sum(
            [len(self.tokenizer(p, return_tensors="pt").input_ids) for p in prompt]
        )
        max_context_length = self.max_tokens - prompt_length - self.max_new_tokens
        return max_context_length

    def prepare_contexts(
        self,
        passages,
        query,
    ):
        """
        prepares the binned contexts
        input: passages (Passages)
               query (str)
        output: contexts (list[str])
        """
        prompt = self.query_prompt.format(query, "")
        max_context_length = self.calculate_max_context_length(prompt)
        sentences = [
            sentence.text
            for passage in passages.get_passages()
            for sentence in passage.passage.sents
        ]
        contexts = segment_text(sentences, self.tokenizer, max_context_length)
        return contexts

    def analyze_contexts(self, contexts):
        context_lengths = []
        for context in contexts:
            context_lengths.append(
                self.tokenizer(context, return_tensors="pt").input_ids.shape[1]
            )
        return context_lengths

    def analyze_passages(self, passages):
        contexts = [passage.passage.text for passage in passages.get_passages()]
        return self.analyze_contexts(contexts)

    def answer_based_on_context(self, context, query):
        """
        answer synthesis
        input: context (str)
               query (str)
        output: answers (list[str])
        """
        # construct full prompt from context and query
        prompted_context = self.query_prompt.format(context, query)
        if not self.ctransformers:
            # initialize inputs
            inputs = self.tokenizer(prompted_context, return_tensors="pt").to(
                self.device
            )
            # generate answers
            output_ids = self.model.generate(
                inputs.input_ids, generation_config=self.generation_config
            )
            # track the start of the answers
            answer_starts = [
                len(inputs.input_ids[i]) for i in range(len(inputs.input_ids))
            ]
            # decode the answers
            answer = [
                self.tokenizer.decode(
                    output_id[j:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for output_id, j in zip(output_ids, answer_starts)
            ]
            # clear CUDA cache
            del inputs, output_ids
            torch.cuda.empty_cache()
            answers = [a.strip() for a in answer]
        else:
            answer = self.pipe(prompted_context, **self.generation_config)
            answers = [a["generated_text"].strip() for a in answer]
        return answers

    def aggregate_answers(self, answers, query):
        """
        aggregates answers into single comprehensive answer
        inputs: answers (list[str])
                query (str)
        output: comprehensive_answer (str)
        """
        # construct full prompt from answers and query
        answer_candidates = "".join(
            [
                '\n\n-From Source {}:\n"{}"'.format(i + 1, answer)
                for i, answer in enumerate(answers)
            ]
        )
        prompted_context = self.aggregation_prompt.format(query, answer_candidates)
        if not self.ctransformers:
            # initialize inputs
            inputs = self.tokenizer(prompted_context, return_tensors="pt").to(
                self.device
            )
            # generate answer
            output_ids = self.model.generate(
                inputs.input_ids, generation_config=self.generation_config
            )
            # track the start of the answer
            answer_start = len(inputs.input_ids[0])
            # decode the answer
            comprehensive_answer = self.tokenizer.decode(
                output_ids[0][answer_start:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            # clear CUDA cache
            del inputs, output_ids
            torch.cuda.empty_cache()
        else:
            comprehensive_answer = self.pipe(
                prompted_context, **self.generation_config
            )[0]["generated_text"].strip()
        return comprehensive_answer

    def validate_context(self, context, query):
        """
        validates context relevancy to query
        WARNING: untested
        inputs: context (str)
                query (str)
        output: answer (str)
        """
        # construct full prompt from context and query
        prompted_context = self.validation_prompt.format(context, query)
        # initialize inputs
        inputs = self.tokenizer(prompted_context, return_tensors="pt").to(self.device)
        # generate answer
        output_ids = self.model.generate(
            inputs.input_ids, generation_config=self.generation_config
        )
        # track the start of the answer
        answer_start = len(inputs.input_ids[0])
        # decode the answer
        answer = self.tokenizer.decode(
            output_ids[0][answer_start:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        # clear CUDA cache
        del inputs, output_ids
        torch.cuda.empty_cache()
        return answer


class ModelHandler:
    """
    handles model loading, unloading and serving
    """

    def __init__(self):
        """
        empty initializer
        """
        pass

    def load_corpus_model(
        self,
    ):
        """
        loads corpus model
        """
        self.corpus_model = CorpusModel()

    def unload_corpus_model(
        self,
    ):
        """
        unloads corpus model
        """
        del self.corpus_model

    def load_embedding_model(
        self,
        model_path,
        device,
    ):
        """
        loads embedding model
        input: model_path (str)
               device (str)
        """
        self.embedding_model = EmbeddingModel(
            model_path,
            device,
        )

    def unload_embedding_model(self):
        """
        unloads embedding model
        """
        del self.embedding_model
        torch.cuda.empty_cache()

    def load_cross_encoding_model(self, model_path, device):
        """
        loads cross-encoding model
        input: model_path (str)
               device (str)
        """
        self.cross_encoding_model = CrossEncodingModel(
            model_path,
            device,
        )

    def unload_cross_encoding_model(self):
        """
        unloads cross-encoding model
        """
        del self.cross_encoding_model
        torch.cuda.empty_cache()

    def load_query_model(
        self,
        model_path,
        model_file,
        max_tokens,
        device,
    ):
        """
        loads query model
        input: model_path (str)
               device (str)
        """
        self.query_model = QueryModel(
            model_path,
            model_file,
            max_tokens,
            device,
        )

    def unload_query_model(self):
        """
        unloads query model
        """
        del self.query_model
        torch.cuda.empty_cache()

    def fetch_corpus_model(self, n_proc, batch_size):
        """
        fetches corpus model
        input: n_proc (int)
               batch_size (int)
        """
        self.corpus_model.set_n_proc(n_proc)
        self.corpus_model.set_batch_size(batch_size)
        return self.corpus_model

    def fetch_embedding_model(self, batch_size):
        """
        fetches embedding model
        input: batch_size (int)
        """
        self.embedding_model.set_batch_size(batch_size)
        return self.embedding_model

    def fetch_cross_encoding_model(self, batch_size):
        """
        fetches cross-encoding model
        input: batch_size (int)
        """
        self.cross_encoding_model.set_batch_size(batch_size)
        return self.cross_encoding_model

    def fetch_query_model(self, max_new_tokens):
        """
        fetches query model
        input: max_new_tokens (int)
        """
        self.query_model.set_config(max_new_tokens)
        return self.query_model
