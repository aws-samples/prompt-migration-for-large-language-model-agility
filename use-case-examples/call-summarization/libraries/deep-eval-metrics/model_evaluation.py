from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from typing import Optional, Union
import boto3
from langchain.llms.bedrock import Bedrock
from botocore.config import Config
from langchain_core.language_models import BaseChatModel
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.chat_completion.retry import retry_with_exponential_backoff


class DeepEvalBaseLLM(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = model_name
        self.model = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass

valid_bedrock_models = [
    "anthropic.claude-v2",
    "amazon.titan-text-express-v1", 
    "amazon.titan-text-lite-v1", 
    "amazon.titan-embed-text-v1", 
    "anthropic.claude-v2:1", 
    "anthropic.claude-3-sonnet-20240229-v1:0", 
    "anthropic.claude-instant-v1"
]

default_bedrock_model = "anthropic.claude-v2"


class BedrockModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_bedrock_models:
                raise ValueError(
                    f"Invalid model. Available Bedrock models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_bedrock_model
            
        self.model = self.load_model()

        super().__init__(model_name, *args, **kwargs)

    def load_model(self, region='us-east-1'):
        model_kwargs_claude = {"temperature": 0, 'max_tokens_to_sample': 2000}
        session_kwargs = {"region_name": region}
        client_kwargs = {**session_kwargs}

        session = boto3.Session(**session_kwargs)

        retry_config = Config(
            region_name='us-east-1',
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        bedrock = session.client(
            service_name='bedrock-runtime',
            config=retry_config,
            **client_kwargs
        )

        return Bedrock(model_id=self.model_name, model_kwargs=model_kwargs_claude, client=bedrock)

    @retry_with_exponential_backoff
    def generate(self, prompt: str) -> str:
        return self.model(prompt)
    
    def a_generate(self, prompt: str) -> str:
        return self.model(prompt)

    def get_model_name(self):
        if self.model_name:
            return self.model_name