from retry import retry
from .text_summary import text_summary
import json
import boto3
import time
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import libraries.iaa.configs as configs

# iaa.text_summary import text_summary
class ReportChunk():
    vector_field = None
    metadata = None
    text = None
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
bedrockClient = boto3.client("bedrock-runtime", region_name=configs.REGION)

credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, configs.REGION, "aoss")

openSearchClient = OpenSearch(
    hosts=[{'host': configs.OPEN_SEARCH_HOST, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20,
    timeout=3000
)


def bulkLoadChunks(chunks, indexId, embeddingsModel):

    records = []
    
    print("Embedding vector for section chunks started")
    for chunk in chunks:
        content = chunk['paragraph']
        if "company_name" in chunk.keys():
            metadata = {
                "company_name": chunk['company_name'],
                "doc_type": chunk['doc_type'],
                "time": chunk['time'],
            }
        else:
            metadata = {}


        record = ReportChunk(
                        text=content,
                        metadata=metadata
                    )

        # print("Embedding vector for section chunks started")
        createVectorEmbeddingWithBedrock(record, indexId, embeddingsModel)
        records.append(record.__dict__.copy())
    print("Embedding vector for section chunks completed")
    CHUNK_SIZE = 10
    print("Adding chunks to index")
    for i in range(0, len(records), CHUNK_SIZE):
        record_chunk = records[i:i+CHUNK_SIZE]
        start_time = time.time()
        # print(f"Adding {len(record_chunk)} to index")
        bulkLoadHelper(
            record_chunk
        )
        delta = time.time() - start_time
        # print(f"Adding finished, {delta:.3} seconds")
    print("Adding chunks to index complete")
    print("Uploading documents in OpenSearch Completed")
    return

def bulkLoadHelper(records):
    success, failed = putBulkOpenSearch(records)
    return

@retry(tries=6, delay=3, backoff=3, max_delay=360)
def putBulkOpenSearch(list):
    # print(f"Putting {len(list)} documents in OpenSearch")
    success, failed = bulk(openSearchClient, list)
    return success, failed

@retry(tries=6, delay=3, backoff=3, max_delay=360)
def createVectorEmbeddingWithBedrock(record: ReportChunk, indexName, embeddingsModel):
    if embeddingsModel == 'amazon.titan-embed-text-v1':
        text = record.text
        if len(text) > 20000:
            text = text_summary.summarize_large_chunks(text)
        payload = {"inputText": f"{text}"}
        body = json.dumps(payload)
        modelId = embeddingsModel
        accept = "application/json"
        contentType = "application/json"

        response = bedrockClient.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())

        embedding = response_body.get("embedding")
        record.vector_field = embedding
        record._index = indexName
    else:
        text = record.text
        if len(text) > 20000:
            text = text_summary.summarize_large_chunks(text)
        # below is fix to remove sequences of single character tokens (cohere tokenizer hits # tokens limit)
        text = text.replace('..', '')
        text = text.replace(',,', '')
        payload = {"texts": [f"{text}"], "input_type": "search_document", "truncate": "END"}
        body = json.dumps(payload)
        modelId = embeddingsModel
        accept = "application/json"
        contentType = "application/json"

        response = bedrockClient.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())


        embedding = response_body.get("embeddings")[0]
        record.vector_field = embedding
        record._index = indexName
