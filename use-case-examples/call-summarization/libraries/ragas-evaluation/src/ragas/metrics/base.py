"""
Q - question
A - answer: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truths: ground truth answer
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import floor

import boto3
from botocore.config import Config
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.llms.bedrock import Bedrock
from ragas.llms import LangchainLLM, llm_factory
from tqdm import tqdm

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.bedrock import BedrockChat


def parse_results(statement_parser, result):
    # Parse Results
    result = [statement_parser.invoke(s) for s in result]
    list_statements = []
    for s in result:
        s = list(s)
        list_statements += [s[0][1]]
    return list_statements


def make_batches(total_size: int, batch_size: int) -> list[range]:
    """
    Take a total size and batch size and return a list of ranges for the batches
    """
    tail = total_size % batch_size
    num_batches = floor(total_size / batch_size)
    batches = [
        range(i, i + batch_size) for i in range(0, batch_size * num_batches, batch_size)
    ]
    if tail != 0:
        batches.append(range(batch_size * num_batches, batch_size * num_batches + tail))

    return batches


EvaluationMode = Enum("EvaluationMode", "qac qa qc gc ga qga")


@dataclass
class Metric(ABC):
    batch_size: int

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def evaluation_mode(self) -> EvaluationMode:
        ...

    @abstractmethod
    def init_model():
        """
        This method will lazy initialize the model.
        """
        ...

    def score(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
    ) -> Dataset:
        scores = []
        verdicts = []
        cm = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(f"ragas_{self.name}", callback_manager=cm) as group:
            for batch in tqdm(self.get_batches(len(dataset))):
                score, verdict_str = self._score_batch(
                    dataset.select(batch), callbacks=group
                )
                scores.extend(score)
                verdicts.extend(verdict_str)

        return dataset.add_column(f"{self.name}", scores).add_column(f"verdicts", verdicts)  # type: ignore

    @abstractmethod
    def _score_batch(
        selfself: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
        callback_group_name: str = "batch",
    ) -> list:
        ...

    def score_single(
        self: t.Self,
        ds_row: dict,
        callbacks: t.Optional[Callbacks] = None,
    ) -> float:
        """
        Score for a single row of dataset
        """
        # TODO: validation check if they are string

        ds = Dataset.from_dict({k: [v] for k, v in ds_row.items()})
        score = self._score_batch(
            ds, callback_group_name=self.name, callbacks=callbacks
        )

        return score[0]

    def get_batches(self, dataset_size: int) -> list[range]:
        return make_batches(dataset_size, self.batch_size)


@dataclass
class MetricWithLLM(Metric):
    """Here we will override raga's default behaviour of using openAI with using Bedrock"""

    # Bedrock Integration - Recent Versions of Boto3
    # ----------------------------------------------------------------------------------------------------
    model_kwargs_claude = {"temperature": 0, "max_tokens_to_sample": 2000}
    session_kwargs = {"region_name": "us-east-1"}
    client_kwargs = {**session_kwargs}

    session = boto3.Session(**session_kwargs)

    retry_config = Config(
        region_name="us-east-1",
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    bedrock = session.client(
        service_name="bedrock-runtime", config=retry_config, **client_kwargs
    )

    llm = Bedrock(
        model_id="anthropic.claude-v2", model_kwargs=model_kwargs_claude, client=bedrock
    )
    # --------------------------------------------------------------------------------------------------

    # llm: LangchainLLM = field(default_factory=llm_factory)
    def init_model(self, model_id="anthropic.claude-v2"):
        self.model_id = model_id

        if isinstance(self.llm, ChatOpenAI) or isinstance(self.llm, OpenAI):
            self.llm.langchain_llm = t.cast(ChatOpenAI, self.llm)
            if self.llm.langchain_llm.openai_api_key == "no-key":
                raise OpenAIKeyNotFound


# Use this as a reference to design new metric classes for this package
class Temp(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15

    def _score_batch(self, dataset):
        pass
