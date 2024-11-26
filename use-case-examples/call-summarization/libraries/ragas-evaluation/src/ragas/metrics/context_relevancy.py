from __future__ import annotations

import logging
import time
import typing as t
from dataclasses import dataclass, field
from typing import List

import pysbd
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


context_relevance_prompt = PromptTemplate.from_template(
    """<instructions>
Given a question and a context, verify if each sentence in the context is directly relevant to the question.
</instructions>

<example_input>
<question>
What is the significance of the Statue of Liberty in New York City?
</question>

<context>
The Statue of Liberty National Monument and Ellis Island Immigration Museum are managed by the National Park Service and are in both New York and New Jersey. They are joined in the harbor by Governors Island National Monument. Historic sites under federal management on Manhattan Island include Stonewall National Monument; Castle Clinton National Monument; Federal Hall National Memorial; Theodore Roosevelt Birthplace National Historic Site; General Grant National Memorial (Grant's Tomb); African Burial Ground National Monument; and Hamilton Grange National Memorial. Hundreds of properties are listed on the National Register of Historic Places or as a National Historic Landmark.
</context>
</example_input>

<example_response>
1. The Statue of Liberty National Monument and Ellis Island Immigration Museum are managed by the National Park Service and are in both New York and New Jersey. [No]
2. They are joined in the harbor by Governors Island National Monument. Historic sites under federal management on Manhattan Island include Stonewall National Monument; Castle Clinton National Monument; Federal Hall National Memorial; Theodore Roosevelt Birthplace National Historic Site; General Grant National Memorial (Grant's Tomb); African Burial Ground National Monument; and Hamilton Grange National Memorial. [Yes]
3. Hundreds of properties are listed on the National Register of Historic Places or as a National Historic Landmark. [Yes]
</example_response>

Here is the question and context for you to analyze:
<question>
{question}
</question>

<context>
{context}
</context>

Remember it is absolutely critical that you do not output double newlines in the response!

Assistant:
Line by line sentence classifications for the given answer:
"""  # noqa: E501
)
CONTEXT_RELEVANCE_PROMPT = HumanMessagePromptTemplate(prompt=context_relevance_prompt)


seg = pysbd.Segmenter(language="en", clean=False)


def sent_tokenize(text: str) -> List[str]:
    """
    tokenizer text into sentences
    """
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences


@dataclass
class ContextRelevancy(MetricWithLLM):
    """
    Extracts sentences from the context that are relevant to the question with
    self-consistancy checks. The number of relevant sentences and is used as the score.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_relevancy"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qc  # type: ignore
    batch_size: int = 15
    show_deprecation_warning: bool = False

    @staticmethod
    def sleep():
        return 10

    def _compute_score(self, response: str) -> float:
        context = "\n".join(row["contexts"])
        context_sents = sent_tokenize(context)
        indices = (
            sent_tokenize(response.strip())
            if response.lower() != "insufficient information."
            else []
        )
        # print(len(indices))
        if len(context_sents) == 0:
            return 0
        else:
            return min(len(indices) / len(context_sents), 1)

    def _score_batch(
        self, dataset: Dataset, callbacks: Callbacks, callback_group_name: str = "batch"
    ) -> float:
        assert self.llm is not None, "LLM is not initialized"

        question, contexts = dataset["question"], dataset["contexts"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            for q, ctx in zip(question, contexts):
                prompt_text = CONTEXT_RELEVANCE_PROMPT.format(
                    question=q, context="\n".join(ctx)
                )
                # human_prompt = PromptTemplate(
                #     template=prompt_text + '\n {statement_parser.get_format_instructions()}',
                #     input_variables=[],
                #     partial_variables={"format_instructions": statement_parser.get_format_instructions()}
                # )
                # human_prompt = PromptTemplate.from_template(prompt_text)
                # human_prompt.partial_variables = {"format_instructions": statement_parser.get_format_instructions()}
                # prompts.append(ChatPromptTemplate(human_prompt))

                temp_prompt = ChatPromptTemplate.from_messages([prompt_text])

                # temp_prompt.partial_variables = {"format_instructions": statement_parser.get_format_instructions()}
                # prompts.append(ChatPromptTemplate.from_messages([human_prompt]))
                prompts.append(temp_prompt)

        responses: list[list[str]] = []
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

        responses = [x.strip().split("\n") for x in results]

        scores = []
        all_verdicts = []
        for i, response in enumerate(responses):
            response = [r for r in response if r]
            numerator = 0
            for r in response:
                if "[Yes]" in r:
                    numerator += 1
            denom = len(response)
            scores += [numerator / denom]
            verdicts_str = response
            all_verdicts += [verdicts_str]
        return scores, all_verdicts


context_relevancy = ContextRelevancy()
