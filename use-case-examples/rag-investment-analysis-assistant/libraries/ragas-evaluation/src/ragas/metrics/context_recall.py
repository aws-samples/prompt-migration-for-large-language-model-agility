from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass

import numpy as np
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

answer_prompt = PromptTemplate.from_template(
    """\
Given a question and answer, create one or more statements from each sentence in the given answer.
<examples>
question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:\nAlbert Einstein was born in Germany.\nAlbert Einstein was best known for his theory of relativity.
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements:\nCadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of the same nationality?
answer: They were from different countries.
statements:\nShahul and Jithin were from different countries.
</examples>
question:{question}
answer: {answer}
statements:\n"""  # noqa: E501
)

LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate(prompt=answer_prompt)

context_recall_ra = PromptTemplate.from_template(
    """<instructions>
Given a context, and an answer, first identify each individual sentence in the answer, then analyze each, and classify if the sentence can be attributed to the given context or not.
Carefully think step by step and provide detailed reasoning reflecting the context provided before coming to conclusion.
Follow the exact output format as shown in the example response. NEVER output double newlines in the response!
</instructions>

Here are some examples with responses:

<example_input>
<context>
Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
</context>

<answer>
Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895
</answer>
</example_input>

<example_response>
Line by line sentence classifications for the given answer:
1. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. The date of birth of Einstein is mentioned clearly in the context. So [Attributed]
2. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. The exact sentence is present in the given context. So [Attributed]
3. He published 4 papers in 1905. There is no mention about papers he wrote in given the context. So [Not Attributed]
4. Einstein moved to Switzerland in 1895. There is not supporting evidence for this in the given the context. So [Not Attributed]
</example_response>

<example_input>
<context>
William Shakespeare (26 April 1564 – 23 April 1616) was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language and the world's greatest dramatist. He is often called England's national poet and the "Bard of Avon". His extant works, including some collaborations, consist of about 39 plays, 154 sonnets, and two long narrative poems. His plays have been translated into every major living language and are performed more often than those of any other playwright. He was born in Stratford-upon-Avon and married Anne Hathaway, with whom he had three children.
</context>

<answer>
William Shakespeare was born on 26 April 1564 in Stratford-upon-Avon. He wrote 39 plays and 154 sonnets. Shakespeare was known as the Bard of Avon. He had two children with Anne Hathaway.
</answer>
</example_input>

<example_response>
Line by line sentence classifications for the given answer:
1. William Shakespeare was born on 26 April 1564 in Stratford-upon-Avon. The date and place of birth are mentioned clearly in the context. So [Attributed]
2. He wrote 39 plays and 154 sonnets. The exact number of plays and sonnets is present in the given context. So [Attributed]
3. Shakespeare was known as the Bard of Avon. This title is specifically mentioned in the context. So [Attributed]
4. He had two children with Anne Hathaway. The context states that he had three children with Anne Hathaway, not two. So [Not Attributed]
</example_response>

Here is the input and answer for you to classify:

<context>
{context}
</context>

<answer>
{ground_truth}
</answer>

Remember it is absolutely critical that you do not output double newlines in the response!

Assistant:
Line by line sentence classifications for the given answer:
"""  # noqa: E501
)

CONTEXT_RECALL_RA = HumanMessagePromptTemplate(prompt=context_recall_ra)


@dataclass
class ContextRecall(MetricWithLLM):

    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_recall"
    evaluation_mode: EvaluationMode = EvaluationMode.gc
    batch_size: int = 15

    @staticmethod
    def sleep():
        return 10

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        verdict_token = "[Attributed]"
        prompts = []
        ground_truths, contexts, question = (
            dataset["ground_truths"],
            dataset["contexts"],
            dataset["question"],
        )

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for q, gt in zip(question, ground_truths):
                prompt_text = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=gt)
                temp_prompt = ChatPromptTemplate.from_messages([prompt_text])
                prompts.append(temp_prompt)

            prompts = []
            list_statements = []
            for gt, ctx in zip(ground_truths, contexts):
                gt = "\n".join(gt) if isinstance(gt, list) else gt
                ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
                human_prompt = CONTEXT_RECALL_RA.format(context=ctx, ground_truth=gt)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            responses: list[list[str]] = []
            try:
                results = self.llm.batch([x.messages for x in prompts])
            except:
                time.sleep(self.sleep())
                results = self.llm.batch([x.messages for x in prompts])

            responses = [x.strip().split("\n") for x in results]

            scores = []
            all_verdicts = []
            for i, response in enumerate(responses):
                response = [r for r in response if r]
                numerator = 0
                for r in response:
                    if verdict_token in r:
                        numerator += 1
                denom = len(response)
                scores += [numerator / denom]
                verdicts_str = response
                all_verdicts += [verdicts_str]

        return scores, all_verdicts


context_recall = ContextRecall()
