from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class HasSegmentMethod(t.Protocol):
    def segment(self, text) -> t.Any: ...


logger = logging.getLogger(__name__)


class AnswerPrecisionStatements(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")
    sentences: t.Dict[int, str] = Field(
        description="A mapping of sentence index to the sentence"
    )


class SentenceComponents(BaseModel):
    sentence_index: int = Field(description="The index of the sentence")
    simpler_statements: t.List[str] = Field(
        description="A list of simpler statements that can be directly inferred from the context"
    )


class SentencesSimplified(BaseModel):
    sentences: t.List[SentenceComponents] = Field(
        description="A list of sentences and their simpler versions"
    )


# examples
example_input_1 = AnswerPrecisionStatements(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    sentences={
        0: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.",
        1: "He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    },
)

example_output_1 = SentencesSimplified(
    sentences=[
        SentenceComponents(
            sentence_index=0,
            simpler_statements=[
                "Albert Einstein was a German-born theoretical physicist.",
                "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
            ],
        ),
        SentenceComponents(
            sentence_index=1,
            simpler_statements=[
                "Albert Einstein was best known for developing the theory of relativity.",
                "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
            ],
        ),
    ]
)


class LongFormAnswerPrompt(PydanticPrompt[AnswerPrecisionStatements, SentencesSimplified]):
    instruction = "Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON."
    input_model = AnswerPrecisionStatements
    output_model = SentencesSimplified
    examples = [(example_input_1, example_output_1)]


class StatementAnswerPrecisionAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementAnswerPrecisionAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context. The statement and reason should be under double quotes. If content within the statement and reason have quotes, make sure it is under single quotes. For example, the format should be: 
    
    "reason": "This is the reason because of this 'quote'
    
    and NOT: 
    "reason": "This is the reason because of this "quote"
    
    """
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementAnswerPrecisionAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementAnswerPrecisionAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementAnswerPrecisionAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                    StatementAnswerPrecisionAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information given in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                statements=[
                    "Albert Einstein was a genius.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementAnswerPrecisionAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="The context and statement are unrelated",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


@dataclass
class AnswerPrecision(MetricWithLLM, SingleTurnMetric):
    name: str = "answer_precision"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts", 
                "reference", 
                "response", 
                "user_input"
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    nli_statements_message: PydanticPrompt = field(default_factory=NLIStatementPrompt)
    statement_prompt: PydanticPrompt = field(default_factory=LongFormAnswerPrompt)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["reference"])
        verdicts = await self.nli_statements_message.generate(
            data=NLIStatementInput(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> SentencesSimplified:
        assert self.llm is not None, "llm is not set"
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        text, question = row["response"], row["user_input"]
        sentences = self.sentence_segmenter.segment(text)
        sentences_with_index = {i: sentence for i, sentence in enumerate(sentences)}

        statements_simplified = await self.statement_prompt.generate(
            llm=self.llm,
            data=AnswerPrecisionStatements(
                question=question, answer=text, sentences=sentences_with_index
            ),
            callbacks=callbacks,
        )
        return statements_simplified

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"
        

        statements_simplified = await self._create_statements(row, callbacks)
        if statements_simplified is None:
            return np.nan

        # unwrap the statements
        statements = []
        for component in statements_simplified.sentences:
            statements.extend(component.simpler_statements)

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)
    
answer_precision = AnswerPrecision()
