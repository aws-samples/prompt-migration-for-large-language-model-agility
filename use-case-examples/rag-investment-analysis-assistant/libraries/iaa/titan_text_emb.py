import base64
import json

import boto3
import numpy as np


class BedrockTextEmbedding:
    region_name: str = "us-east-1"
    fmc_url: str = "https://bedrock-runtime.us-east-1.amazonaws.com"
    service_name: str = "bedrock-runtime"

    def initialize_model(self):
        bedrock = boto3.client(
            service_name=self.service_name,
            region_name=self.region_name,
            endpoint_url=self.fmc_url,
        )
        print("Bedrock client loaded")
        self.bedrock = bedrock

    def get_embeddings(self, inputText, **kwargs):
        modelId = "amazon.titan-embed-text-v1"
        accept = "application/json"
        contentType = "application/json"

        body = json.dumps({"inputText": inputText})

        response = self.bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

        response_body = json.loads(response.get("body").read())
        emb = response_body.get("embedding")

        return np.array(emb)

    def cosine_similarity(self, source_emb, target_emb):
        source_emb = np.array(source_emb) / np.linalg.norm(source_emb)
        target_emb = np.array(target_emb) / np.linalg.norm(target_emb)

        return np.dot(source_emb, target_emb)
