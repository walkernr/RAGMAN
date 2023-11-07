import os
from termcolor import colored, cprint
import ujson as json
import hashlib
from ragman import RAGMAN

LOG_DIR = "./.logs"


class Session:
    def __init__(self):
        self.width = os.get_terminal_size().columns
        self.ragman = RAGMAN()

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
                        cprint(sentence.text, "light_red", end=" ")
                    elif passage.score[j] >= 0.25:
                        cprint(sentence.text, "yellow", end=" ")
                    else:
                        cprint(sentence.text, "light_grey", end=" ")
                else:
                    cprint(sentence.text, "light_cyan", end=" ")
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
            cprint("Invalid selection.", "light_red")
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
            cprint("Invalid selection.", "light_red")
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
            cprint("Invalid selection.", "light_red")
            return None

    def query_corpus(self, corpus):
        if not corpus:
            cprint(
                "No corpus selected. Please build, load, or select a corpus.", "yellow"
            )
            return "B"

        while True:
            print("Selected corpus: {}".format(corpus))
            query = input("Please enter a query (B to go back and Q to quit): ").lower()

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
                cprint("Invalid selection.", "light_red")


if __name__ == "__main__":
    session = Session()
    session.run()
