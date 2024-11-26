import json

import boto3
from typing import List
from langchain.llms.bedrock import Bedrock

import libraries.iaa.configs as configs


class GenerateAnswer:
    """
    This class uses the prompt PROMPT_QA defined in config file to generate
    answer using the context retrieved and the query.
    """
    def __init__(self):
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=configs.REGION)

    def format_prompt(self, question, time_kwds, context, rephrased_query):
        """Format the final prompt to pass to LLM for question answering.

        Args:
            question (str): The original input question
            time_kwds (List[str]): List of time related keywords
            context (str): The concatenated top contexts
            rephrased_query (str): Rephrased version of original question

        Returns:
            prompt (str): The formatted prompt string
        """
        query = question
        return configs.PROMPT_QA.format(
            query=query,
            context=context,
            time_kwds=time_kwds,
            rephrased_query=rephrased_query,
            most_recent_quarter=configs.MOST_RECENT_QUARTER,
        )

    def format_prompt_v2(self, question: str, time_kwds: List[str], context:str, rephrased_query: str):
        """Format the final prompt to pass to LLM for question answering.

        Args:
            question (str): The original input question
            time_kwds (List[str]): List of time related keywords
            context (str): The concatenated top contexts
            rephrased_query (str): Rephrased version of original question

        Returns:
            prompt (str): The formatted prompt string
        """
        query = question
        return configs.PROMPT_QA_v2.format(
            query=query,
            context=context,
            time_kwds=time_kwds,
            rephrased_query=rephrased_query,
            most_recent_quarter=configs.MOST_RECENT_QUARTER,
        )

    def get_llm_answer_stream(self, model_id: str, max_tokens: int, temperature: float, prompt: str):
        """Use Bedrock Streaming API invoke_model_with_response_stream to generate answer
        
        Args:
            model_id: LLM model to use.
            max_tokens: Max tokens to generate
            temperature: Parameter that controls the randomness of the generated output
            prompt: Prompt to be used for answer generation

        Returns:
            answer (str): The answer based on context and query
        """
        body = json.dumps(
            {
                "prompt": prompt,
                "top_k": 50,
                "top_p": 0.1,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "stop_sequences": ["Question"],
            }
        )
        pred = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id,
            body=body,
        )
        return pred

    def get_llm_answer(self, model_id: str, max_tokens: int = 800, temperature: float = 0.1, prompt: str = None):
        """Initialize and return LLM model.

        Args:
            model_id: ID of model to use
            max_tokens: Maximum number of tokens
            temperature: Sampling temperature

        Returns:
            llm: Initialized LLM instance
        """
        if model_id != "anthropic.claude-3-haiku-20240307-v1:0":
            body = json.dumps({
                "prompt": prompt,
                "top_k": 250,
                "top_p": 0.5,
                "max_tokens_to_sample": max_tokens,
                "temperature": 0,
                "stop_sequences": ["Question"]
            })
            response = self.bedrock_runtime.invoke_model(
                modelId = 'anthropic.claude-instant-v1',
                accept = 'application/json',
                contentType = 'application/json',
                body=body
            )
            response_body = json.loads(response.get('body').read())
            return response_body.get('completion')
        else:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages":[{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": prompt
                        }]
                    }],
                    "top_k": 50,
                    "top_p": 0.1,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop_sequences": ["Human"],
                }
            )
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                accept = 'application/json',
                contentType = 'application/json'
            )

            response_body = json.loads(response.get('body').read())
            # print(response_body)
            # text
            return response_body.get('content')[0]['text']