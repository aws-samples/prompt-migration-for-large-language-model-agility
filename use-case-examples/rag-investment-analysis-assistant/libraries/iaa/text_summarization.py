import base64
import json

import boto3
import numpy as np
from bs4 import BeautifulSoup
import libraries.iaa.configs as configs
zero_shot_summarization = """
\n\nHuman:
<instructions>
Human: You are a financial document summarization bot.
Given a paragraph, summarize the entire paragraph into less than 200 words. The paragraph can contain csv tables. Summarize them too.
</instructions>

<FORMATTING>
Return the output with the following format
</summary>
Add the summary of the paragraph here.
<summary>
</FORMATTING>

< INPUT >
{inputs}
</INPUT>

<< OUTPUT (must <summary> tag) >>
<< OUTPUT (must end with </summary> tag) >>
\n\nAssistant:
"""

zero_shot_keyword = """
\n\nHuman:
<instructions>
Human: You are a keyword extraction bot.
Given a conversation identify important keywords and speakers.
</instructions>

<FORMATTING>
Return the output with the following format
<keywords>
Extracts key entities and keywords from the conversation here
</keywords>
<speakers>
Extracts speaker names here (comma separated)
If there are no speakers identified, return empty string
</speakers>
</FORMATTING>

< INPUT >
{inputs}
</INPUT>

<< OUTPUT (must have <keyword><speakers> tag) >>
<< OUTPUT (must end with </keyword> and </speakers> tag) >>
\n\nAssistant:
"""


class BedrockTextGeneration:
    region_name: str = configs.REGION
    fmc_url: str = "https://bedrock-runtime.us-west-2.amazonaws.com" #bedrock-runtime.us-west-2.amazonaws.com , bedrock-runtime.us-west-2.amazonaws.com
    service_name: str = "bedrock-runtime"

    def initialize_model(self):
        bedrock = boto3.client(
            service_name=self.service_name,
            region_name=self.region_name,
            endpoint_url=self.fmc_url,
        )
        print("Bedrock client loaded")
        self.bedrock = bedrock

    def get_generation(self, inputText, zero_shot=True):
        """Generate bedrock summary of long transcripts

        Args:
            inputText (str): transcribe script
            zero_shot (bool, optional): Use zeroshot prompt. Defaults to True.
            summarization (bool, optional): Use summarization function. Defaults to True.
            kw_extraction (bool, optional): Use keyword extraction. Defaults to False.

        Returns:
            str: outputText string
        """
        accept = "application/json"
        contentType = "application/json"
        model_id = "anthropic.claude-v2"

        prompt = zero_shot_summarization.format(inputs=inputText)

        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens_to_sample": 4096,
                "temperature": 0.0,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"],
            }
        )
        response = self.bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
        response_body = json.loads(response.get("body").read())
        outputText = response_body.get("completion")
        return outputText

    def cosine_similarity(self, source_emb, target_emb):
        source_emb = np.array(source_emb) / np.linalg.norm(source_emb)
        target_emb = np.array(target_emb) / np.linalg.norm(target_emb)
        return np.dot(source_emb, target_emb)


def parse_llm_output(output_string):
    """
    Given bedrock summarization of transcription, extract summary, keyword and speakers based on XML tags
    """
    soup = BeautifulSoup(output_string, "html.parser")
    summary = soup.find("summary".lower())
    summary = summary.get_text().strip() if summary else ""

    return summary
