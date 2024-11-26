from __future__ import annotations

import os
import time
import typing as t
from dataclasses import dataclass
from enum import Enum

import boto3
import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.embeddings import BedrockEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel

# from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.base import MetricWithLLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager


class QuestionGenFormat(BaseModel):
    pass


QUESTION_GEN = HumanMessagePromptTemplate.from_template(
    """
Generate question for the given answer.
<examples>
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
</examples>
Answer:{answer}
Question:
"""  # noqa: E501
)
EvaluationMode = Enum("EvaluationMode", "qac qa qc gc ga qga")


@dataclass
class AnswerRelevancy(MetricWithLLM):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qa
    batch_size: int = 15
    strictness: int = 3
    embeddings: Embeddings | None = None

    # We will have to replace this embeddings with bedrock or transformer embeddings

    @staticmethod
    def get_embedding_model(model_id: str = "amazon.titan-embed-text-v1"):
        # return SentenceTransformer('sentence-transformers/gtr-t5-large') to use sentence transformer
        # Initialize bedrock client
        bedrock_client = boto3.client("bedrock-runtime")
        return BedrockEmbeddings(client=bedrock_client, model_id=model_id)
    
    @staticmethod
    def sleep():
        return 10

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            # self.embeddings = OpenAIEmbeddings(openai_api_key=oai_key)  # type: ignore
            self.embeddings = self.get_embedding_model()

    def init_model(self):
        super().init_model()

    def score(self: t.Self, dataset: Dataset, callbacks: t.Optional[Callbacks] = None):
        "Overridden function to return a point estimate"
        scores = []
        cm = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(f"ragas_{self.name}", callback_manager=cm) as group:
            for batch in tqdm(self.get_batches(len(dataset))):
                score = self._score_batch(dataset.select(batch), callbacks=group)
                scores.extend(score)

        return dataset.add_column(f"{self.name}", scores)  # type: ignore

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        questions, answers = dataset["question"], dataset["answer"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            for ans in answers:
                human_prompt = QUESTION_GEN.format(answer=ans)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            results = None
            attempt = 0
            while attempt < 2:
                try:
                    results = self.llm.batch([x.messages for x in prompts])
                    break
                except Exception as err:
                    print(err)
                attempt += 1
                time.sleep(self.sleep())

            # results = self.llm.generate(
            #     prompts,
            #     n=self.strictness,
            #     callbacks=batch_group,
            # )
            # results = [[i.text for i in r] for r in results.generations]

            scores = []
            for question, gen_questions in zip(questions, results):
                gen_questions = gen_questions  # .content # Extracting the generated question from object
                cosine_sim = self.calculate_similarity(question, gen_questions)
                scores.append(cosine_sim.mean())

        return scores

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        # question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_query(generated_questions)
        ).reshape(1, -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )


answer_relevancy = AnswerRelevancy()
