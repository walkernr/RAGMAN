import os
import platform
from termcolor import colored, cprint
import ujson as json
import hashlib
from ragman import RAGMAN

LOG_DIR = "./.logs"


class Session:
    def __init__(self):
        self.width = os.get_terminal_size().columns
        self.ragman = RAGMAN(
            retrieval_config=[
                {
                    "name": "pool",
                    "parameters": {
                        "pooling": "arithmetic_mean",
                        "k": 10,
                        "retriever_config": [
                            {
                                "name": "bm25",
                                "parameters": {
                                    "keyword_k": -3,
                                    "k": 1000,
                                },
                                "weight": 0.2,
                            },
                            {
                                "name": "vector",
                                "parameters": {
                                    "k": 1000,
                                },
                                "weight": 0.8,
                            },
                        ],
                    },
                },
                {
                    "name": "cross_encoder",
                    "parameters": {
                        "passage_search": True,
                        "pooling": "max",
                        "k": 0.75,
                    },
                },
            ],
            embedding_model_path="BAAI/bge-large-en-v1.5",
            cross_encoding_model_path="BAAI/bge-reranker-large",
            query_model_path=(
                "TheBloke/neural-chat-7B-v3-3-GGUF"
                if platform.system() == "Darwin"
                else "TheBloke/neural-chat-7B-v3-3-GPTQ"
            ),
            query_model_file=(
                "neural-chat-7b-v3-3.Q4_K_M.gguf"
                if platform.system() == "Darwin"
                else None
            ),
            validate_retrieval=False,
            n_proc=1,
            corpus_processing_batch_size=10000,
            corpus_encoding_batch_size=32,
            ranking_batch_size=32,
            max_tokens=4096,
            max_new_tokens=512,
            retrieval_device="mps" if platform.system() == "Darwin" else "cpu",
            query_device="mps" if platform.system() == "Darwin" else "cuda:0",
        )

    def print_context(self, passages):
        for passage in passages.get_passages():
            cprint("Document: {}".format(passage.document_id), "cyan")
            cprint("Page: {}".format(passage.document_passage_id + 1), "cyan")
            cprint("Top Score: {}".format(passage.score[0]), "cyan")
            for i, sentence in enumerate(passage.passage.sents):
                if i in passage.passage_sentence_id:
                    j = passage.passage_sentence_id.index(i)
                    if passage.score[j] >= 0.75:
                        cprint(sentence.text, "green", end=" ")
                    elif passage.score[j] >= 0.5:
                        cprint(sentence.text, "blue", end=" ")
                    elif passage.score[j] >= 0.25:
                        cprint(sentence.text, "yellow", end=" ")
                    else:
                        cprint(sentence.text, "red", end=" ")
                else:
                    cprint(sentence.text, "white", end=" ")
            print("\n")

    @staticmethod
    def get_user_choice(choices, width, prompt="Selection: "):
        for idx, choice in enumerate(choices, start=1):
            cprint("{}) {}".format(idx, choice), "blue")
        cprint("B) Back", "cyan")
        cprint("Q) Exit", "red")
        print(width * "-")
        selection = input(prompt).lower()
        return selection

    def build_corpus(self):
        document_sets = os.listdir("./documents")
        cprint("Please select a document set to build.", "green")
        selection = self.get_user_choice(document_sets, self.width)

        if selection.lower() in ["b", "q"]:
            return selection

        try:
            idx = int(selection) - 1
            document_set = document_sets[idx]
            print(self.width * "-")
            cprint("Building corpus...", "green")
            print(self.width * "-")
            self.ragman.build_corpus(document_set)
            print(self.width * "-")
            cprint("Loading corpus...", "green")
            self.ragman.load_corpus(document_set)
            return document_set
        except:
            cprint("Invalid selection.", "red")
            return None

    def load_corpus(self):
        corpi = os.listdir("./.corpus")
        if not corpi:
            cprint("No corpus built. Please build a corpus.", "yellow")
            return None

        cprint("Please select a corpus to load.", "green")
        selection = self.get_user_choice(corpi, self.width)

        if selection in ["b", "q"]:
            return selection

        try:
            idx = int(selection) - 1
            corpus = corpi[idx]
            print(self.width * "-")
            cprint("Loading corpus...", "green")
            self.ragman.load_corpus(corpus)
            return corpus
        except:
            cprint("Invalid selection.", "red")
            return None

    def select_corpus(self):
        corpi = list(self.ragman.corpus.keys())
        if not corpi:
            cprint("No corpus loaded. Please build or load a corpus.", "yellow")
            return None

        cprint("Please select a corpus to query.", "green")
        selection = self.get_user_choice(corpi, self.width)

        if selection in ["b", "q"]:
            return selection

        try:
            idx = int(selection) - 1
            return corpi[idx]
        except:
            cprint("Invalid selection.", "red")
            return None

    def query_corpus(self, corpus):
        if not corpus:
            cprint(
                "No corpus selected. Please build, load, or select a corpus.", "yellow"
            )
            return "B"

        while True:
            print("Selected corpus: {}".format(corpus))
            query = input("Please enter a query (B to go back and Q to quit): ")

            if query.lower() == "b":
                return "B"
            elif query.lower() == "q":
                return "Q"
            print(self.width * "-")
            print("Searching...")
            print(self.width * "-")
            passages, answer = self.ragman.search_answer(corpus, query)
            print(self.width * "-")
            cprint("Context:", "green")
            self.print_context(passages)
            print(self.width * "-")
            cprint("Query: {}".format(query), "blue")
            cprint("Answer: {}".format(answer["aggregate_answer"]), "green")
            print(self.width * "-")
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            with open(
                os.path.join(LOG_DIR, "{}_{}.json".format(corpus, query_hash)), "w"
            ) as f:
                json.dump(
                    {
                        "generation": answer,
                        "retrieval": passages.get_serializable_passages(),
                    },
                    f,
                    indent=2,
                )

    def run(self):
        print(self.width * "=")
        cprint("Welcome to RAGMAN.", "green")
        print(self.width * "=")
        current_corpus = None
        while True:
            print(self.width * "-")
            cprint("Please select an action:", "green")
            cprint("1) Build corpus", "blue")
            cprint("2) Load corpus", "blue")
            cprint("3) Select corpus", "blue")
            cprint("4) Query corpus", "blue")
            cprint("Q) Exit", "red")
            print(self.width * "-")

            action = input("Selection: ")
            print(self.width * "-")

            if action == "1":
                new_corpus = self.build_corpus()
                if new_corpus is not None:
                    current_corpus = new_corpus
            elif action == "2":
                new_corpus = self.load_corpus()
                if new_corpus is not None:
                    current_corpus = new_corpus
            elif action == "3":
                new_corpus = self.select_corpus()
                if new_corpus is not None:
                    current_corpus = new_corpus
            elif action == "4":
                result = self.query_corpus(current_corpus)
                if result.lower() == "q":
                    cprint("Goodbye.", "green")
                    break
            elif action.lower() == "q":
                cprint("Goodbye.", "green")
                break
            else:
                cprint("Invalid selection.", "red")


if __name__ == "__main__":
    session = Session()
    session.run()
