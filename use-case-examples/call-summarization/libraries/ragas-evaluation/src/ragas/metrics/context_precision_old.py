from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass
from itertools import combinations, product
from typing import List

import numpy as np
import pysbd
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from pydantic import BaseModel, Field, validator
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from sentence_transformers import CrossEncoder


class StatementFormat(BaseModel):
    statements: List[str] = Field(
        description="List of one element signifying Yes or No response"
    )


statement_parser = PydanticOutputParser(pydantic_object=StatementFormat)

context_precision_prompt = PromptTemplate.from_template(
    """\
<instructions>
Given a question and a context, verify if the information in the given context is useful in answering the question.
Answer only with a single word of either "Yes" or "No" and nothing else.
</instructions>

<example_input>
<question>
What is the significance of the Statue of Liberty in New York City?
</question>

<context>
The Statue of Liberty National Monument and Ellis Island Immigration Museum are managed by the National Park Service and are in both New York and New Jersey. They are joined in the harbor by Governors Island National Monument. Historic sites under federal management on Manhattan Island include Stonewall National Monument; Castle Clinton National Monument; Federal Hall National Memorial; Theodore Roosevelt Birthplace National Historic Site; General Grant National Memorial (Grant's Tomb); African Burial Ground National Monument; and Hamilton Grange National Memorial. Hundreds of properties are listed on the National Register of Historic Places or as a National Historic Landmark.
</context>
</example_input>

<example_response>Yes</example_response>

Here is the question and context for you to analyze:
<question>
{question}
</question>

<context>
{context}
</context>

Assistant: The single word answer is:"""  # noqa: E501
    + "\n {format_instructions}"
)

context_precision_prompt.partial_variables = {
    "format_instructions": statement_parser.get_format_instructions()
}

CONTEXT_PRECISION = HumanMessagePromptTemplate(prompt=context_precision_prompt)

seg = pysbd.Segmenter(language="en", clean=False)


def sent_tokenize(text: str) -> List[str]:
    """
    tokenizer text into sentences
    """
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences


class SentenceAgreement:
    def __init__(
        self: t.Self,
        model_name: str = "cross-encoder/stsb-TinyBERT-L-4",
        metric: str = "bert_score",
    ):
        self.metric = metric
        self.cross_encoder = CrossEncoder(model_name)

    def bert_score(self, para1: str, para2: str) -> float:
        sentences1, sentences2 = sent_tokenize(para1), sent_tokenize(para2)
        scores = self.cross_encoder.predict(
            list(product(sentences1, sentences2)), convert_to_numpy=True  # type: ignore
        )
        assert isinstance(scores, np.ndarray), "Expects ndarray"
        scores = scores.reshape(len(sentences1), len(sentences2))
        return scores.max(axis=1).mean()

    @staticmethod
    def jaccard_score(para1: str, para2: str) -> float:
        sentences1, sentences2 = sent_tokenize(para1), sent_tokenize(para2)
        intersect = len(np.intersect1d(sentences1, sentences2))
        union = len(np.union1d(sentences1, sentences2))
        return intersect / union

    def evaluate(self, answers: List[str]) -> np.float_:
        """
        eval nC2 combinations
        """
        scores = []
        groups = combinations(answers, 2)
        for group in groups:
            if self.metric == "jaccard":
                score = self.jaccard_score(*group)  # type: ignore
            elif self.metric == "bert_score":
                score = self.bert_score(*group)  # type: ignore
            else:
                score = 0
                raise ValueError(f"Metric {self.metric} unavailable")
            scores.append(score)
        score = np.mean(scores)
        return score


@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Extracts sentences from the context that are relevant to the question with
    self-consistancy checks. The number of relevant sentences and is used as the score.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    strictness : int
        Controls the number of times sentence extraction is performed to quantify
        uncertainty from the LLM. Defaults to 1.
    agreement_metric : str
        "bert_score" or "jaccard_score", used to measure agreement between multiple
        samples.
    model_name : str
        any encoder model. Used for calculating bert_score.
    """

    name: str = "context_precision"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15
    strictness: int = 1
    agreement_metric: str = "bert_score"
    model_name: str = "cross-encoder/stsb-TinyBERT-L-4"
    show_deprecation_warning: bool = False

    def __post_init__(self: t.Self):
        if self.agreement_metric == "bert_score" and self.model_name is None:
            raise ValueError(
                "model_name must be provided when agreement_metric is bert_score"
            )

    def init_model(self: t.Self):
        super().init_model()
        self.sent_agreement = SentenceAgreement(
            model_name=self.model_name, metric=self.agreement_metric
        )

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        if self.show_deprecation_warning:
            logging.warning(
                "The 'context_relevancy' metric is going to be deprecated soon! Please use the 'context_precision' metric instead. It is a drop-in replacement just a simple search and replace should work."  # noqa
            )
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for q, c in zip(questions, contexts):
                human_prompt = CONTEXT_PRECISION.format(
                    question=q, context="\n".join(c)
                )
                temp_prompt = ChatPromptTemplate.from_messages([human_prompt])
                prompts.append(temp_prompt)

            responses: list[list[str]] = []
            result = self.llm.batch([x.messages for x in prompts])

            for x in result:
                if x.strip().lower() == "insufficient information":
                    responses += ["insufficient information"]
                else:
                    responses += [statement_parser.parse(x).statements]

            scores = []
            all_verdicts = []
            for context, n_response in zip(contexts, responses):
                context = "\n".join(context)
                overlap_scores = []
                context_sents = sent_tokenize(context)
                for output in n_response:
                    indices = (
                        sent_tokenize(output.strip())
                        if output.lower() != "insufficient information."
                        else []
                    )
                    if len(context_sents) == 0:
                        score = 0
                    else:
                        score = min(len(indices) / len(context_sents), 1)
                    overlap_scores.append(score)
                if self.strictness > 1:
                    agr_score = self.sent_agreement.evaluate(n_response)
                else:
                    agr_score = 1
                scores.append(agr_score * np.mean(overlap_scores))
                verdicts = "".join(n_response)
                all_verdicts.append(verdicts)
        return scores, all_verdicts


@dataclass
class ContextRelevancy(ContextPrecision):
    name: str = "context_relevancy"
    show_deprecation_warning: bool = True


context_precision = ContextPrecision()
context_relevancy = ContextRelevancy()
