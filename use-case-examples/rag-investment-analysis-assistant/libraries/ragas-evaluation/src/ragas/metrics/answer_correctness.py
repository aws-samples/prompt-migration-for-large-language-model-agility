from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from ragas.metrics.answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.faithfulness import Faithfulness
from tqdm import tqdm


@dataclass
class AnswerCorrectness(MetricWithLLM):

    """
    Measures answer correctness compared to ground truth as a combination of
    semantic similarity and factuality

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    weights:
        a list of two weights corresponding to semantic similarity and factuality
        Defaults [0.5, 0.5]
    answer_similarity:
        The AnswerSimilarity object
    faithfulness
        The faithfulness object
    """

    name: str = "answer_correctness"
    evaluation_mode: EvaluationMode = EvaluationMode.qga
    batch_size: int = 15
    weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    answer_similarity: AnswerSimilarity | None = None
    faithfulness: Faithfulness | None = None

    def __post_init__(self: t.Self):
        if self.answer_similarity is None:
            self.answer_similarity = AnswerSimilarity(batch_size=self.batch_size)
        if self.faithfulness is None:
            self.faithfulness = Faithfulness(batch_size=self.batch_size)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        if "contexts" in dataset.column_names:
            ds_faithfulness = dataset.remove_columns(["contexts"])
        else:
            ds_faithfulness = dataset

        ds_faithfulness = ds_faithfulness.rename_columns({"ground_truths": "contexts"})
        faith_scores = self.faithfulness._score_batch(ds_faithfulness)[0]  # type: ignore
        similarity_scores = self.answer_similarity._score_batch(dataset)  # type: ignore

        scores_stacked = np.vstack([faith_scores, similarity_scores])
        scores = np.average(
            scores_stacked,
            axis=0,
            weights=self.weights,
        )

        return scores.tolist()

    def score(self: t.Self, dataset: Dataset, callbacks: t.Optional[Callbacks] = None):
        "Overridden function to return a point estimate"
        scores = []
        cm = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(f"ragas_{self.name}", callback_manager=cm) as group:
            for batch in tqdm(self.get_batches(len(dataset))):
                score = self._score_batch(dataset.select(batch), callbacks=group)
                scores.extend(score)

        return dataset.add_column(f"{self.name}", scores)  # type: ignore


answer_correctness = AnswerCorrectness()
