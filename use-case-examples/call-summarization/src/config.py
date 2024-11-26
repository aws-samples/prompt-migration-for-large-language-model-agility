import boto3

# LLM Parameters and variables
parameters = {"temperature": 0.5}
MAX_TOKENS = 256

my_session = boto3.session.Session()
AWS_REGION = my_session.region_name

OPENAI_MODEL_ID = "gpt-3.5-turbo"
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
MISTRAL_MODEL_ID = "mistral.mistral-large-2402-v1:0"
LAMA_MODEL_ID = "meta.llama3-8b-instruct-v1:0"