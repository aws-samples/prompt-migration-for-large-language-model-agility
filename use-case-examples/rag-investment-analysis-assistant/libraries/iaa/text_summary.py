import json
from boto3 import Session
from botocore.config import Config
import os

import libraries.iaa.configs as configs

class text_summary:
    region_name: str = configs.REGION
    service_name: str = "bedrock-runtime"
    modelId="anthropic.claude-3-haiku-20240307-v1:0"
    @classmethod
    def summarize_large_chunks(self, large_chunks):
        session = Session(region_name=self.region_name)
        bedrock_client = session.client(self.service_name)
        # Prepare the request payload
        prompt = "Extract all information and then summarize all the information extracted without any introductory text or commentary."
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 6000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "text",
                            "text": large_chunks
                        },
                        {
                            "type": "text",
                            "text": "Remove LLM introductory text or commentary"
                        }
                    ]
                }
            ]
        }

        # Invoke the Claude 2 model
        response = bedrock_client.invoke_model(
            modelId=self.modelId,
            contentType="application/json",
            body=json.dumps(payload)
        )

        # Process the response and extract the summary
        summary = json.loads(response['body'].read().decode('utf-8'))['content'][0]['text']

        return summary