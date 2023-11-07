from tqdm import tqdm
from core.model_handler import clean_text

#####################################################################
#### This module handles everything related to answering a query ####
#####################################################################


class QueryHandler:
    def __init__(self):
        pass

    def generate_responses(
        self,
        passages,
        query,
        model,
    ):
        answers = []
        contexts = model.prepare_contexts(passages, query)
        for context in tqdm(contexts, desc="Generating Answers Over Contexts"):
            answer = model.answer_based_on_context(context, query)
            answers.append(answer)
        collated_answers = []
        for context, answer in zip(contexts, answers):
            collated_answers.append(
                {
                    "answer": answer,
                    "context": context,
                }
            )
        return collated_answers

    def generate_aggregate_responses(
        self,
        answers,
        query,
        model,
    ):
        context_answers = []
        for answer in answers:
            if len(answer["answer"]) > 1:
                print("Aggregating In-Context Answers")
                context_answer = model.aggregate_answers(answers, query)
                context_answers.append(
                    {
                        "aggregate_answer": context_answer,
                        "answer": answer["answer"],
                        "context": answer["context"],
                    }
                )
            else:
                context_answers.append(
                    {
                        "aggregate_answer": answer["answer"][0],
                        "answer": answer["answer"],
                        "context": answer["context"],
                    }
                )
        if len(context_answers) > 1:
            print("Aggregating Across-Context Answers")
            across_context_answers = [a["aggregate_answer"] for a in context_answers]
            comprehensive_answer = model.aggregate_answers(
                across_context_answers,
                query,
            )
        else:
            comprehensive_answer = context_answers[0]["aggregate_answer"]
        return {
            "query": query,
            "aggregate_answer": comprehensive_answer,
            "context": context_answers,
        }

    def answer(
        self,
        passages,
        query,
        model,
    ):
        query = clean_text(query).replace("\n", " ").strip()
        answers = self.generate_responses(
            passages,
            query,
            model,
        )
        aggregate_answer = self.generate_aggregate_responses(
            answers,
            query,
            model,
        )
        return {
            "query": aggregate_answer["query"],
            "aggregate_answer": aggregate_answer["aggregate_answer"],
            "context": aggregate_answer["context"],
        }
