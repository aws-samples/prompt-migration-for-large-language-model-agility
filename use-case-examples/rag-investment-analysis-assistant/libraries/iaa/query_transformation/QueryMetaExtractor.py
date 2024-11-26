import json
import re

import boto3
import pandas as pd
from bs4 import BeautifulSoup
from langchain.llms.bedrock import Bedrock

import libraries.iaa.configs as configs


class QueryMetaExtractor:
    """
    This class is used to help with generating metadata in the Q&A pipeline.
    This class helps generates time keywords, query keywords, document type and 
    company name for a given query
    
    Args:
        most_recent_quarter: The current quarter, Default:"Q1'24"
        most_recent_year: The current year, Default: 2024
    """
    def __init__(
        self,
        most_recent_quarter="Q1'24",
        most_recent_year=2024,
    ):
        self.most_recent_quarter = most_recent_quarter
        self.most_recent_year = most_recent_year
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    def get_llm(self, model_id, max_tokens=1000, temperature=0, region="us-east-1", top_p=0.5, top_k=250):
        """
        Create Langchain Bedrock LLM
        Args:
            model_id: Name of the model to be used from bedrock
            max_tokens: The max number of tokens to be generated in response
            temperature: parameter that controls the randomness of the generated output
            region: The AWS region Bedrock should be accessed from
            top_p: known as nucleus sampling or probabilistic sampling, determines the 
                cumulative probability distribution used for sampling the next token in the 
                generated response
            top_k: Sample from the k most likely next tokens at each step. Lower k focuses 
                on higher probability tokens.
        Returns:
            Instance of Bedrock LLM
        """
        BEDROCK = boto3.client("bedrock-runtime", region_name=region)
        
        if model_id == "anthropic.claude-3-haiku-20240307-v1:0":
            return BEDROCK
        else:
            llm = Bedrock(
                region_name=region,
                model_id=model_id,
                model_kwargs={
                    "top_k": top_k,
                    "top_p": top_p,
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature,
                    "stop_sequences": ["Question"],
                },
            )
            llm.client = BEDROCK
            return llm

    def generate_meta(self, llm, prompt, query, time_kwds, use_doc_type):
        """
        Generate Query based keyword, company name and rephrased query Metadata.
        
        Args:
            llm: Bedrock LLM to be used.
            prompt: Prompt to be used to expand query
            query: Question as by user
            time_kwds: Keywords generated from time prompt
            use_doc_type: Boolean to use doc type generation.
        Returns:
            Tuple - rephrased query, query keywords, time kwds, company names, doc type
        """
        prompt_format = prompt.format(query=query, most_recent_quarter=self.most_recent_quarter, time_kwds=time_kwds)
        body = json.dumps({
            # "prompt": prompt_format,
            "anthropic_version": "bedrock-2023-05-31",
            "messages":[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt_format
                }]
            }],
            "top_k": 250,
            "top_p": 0.5,
            "max_tokens": 512,
            "temperature": 0,
            "stop_sequences": ["Human"]
        })
        bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        response = bedrock_client.invoke_model(
            modelId = "anthropic.claude-3-haiku-20240307-v1:0",
            accept = 'application/json',
            contentType = 'application/json',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        llm_output = response_body.get('content')[0]['text']
        
        # llm_output = llm(
        #     prompt.format(query=query, most_recent_quarter=self.most_recent_quarter, time_kwds=time_kwds)
        # )

        rephrased_q, kwds, company_keywords = self.llm_ouput_to_json(llm_output)
        time_kwds = time_kwds.split(",") if type(time_kwds) == str else time_kwds
        time_kwds = [time_kwd.strip() for time_kwd in time_kwds]

        new_time_kwds = []
        for time_kwd in time_kwds:
            new_time_kwd = self.convert_quarter_format(time_kwd)
            if new_time_kwd:
                new_time_kwds.append(new_time_kwd)
        time_kwds.extend(new_time_kwds)
        time_kwds = list(set(time_kwds))

        kwds = list(set(kwds))
        if use_doc_type:
            doc_type = self.doc_classification(time_kwds, rephrased_q)
        else:
            doc_type = []

        if len(time_kwds) == 0 or time_kwds == [""] or time_kwds == "none" or time_kwds is None:
            time_kwds = self.get_last_3_years(company_keywords, doc_type)

        return rephrased_q, kwds, time_kwds, company_keywords, doc_type

    def llm_ouput_to_json(self, llm_output):
        # Use regular expressions to find content between curly braces
        pattern = r"\{([^}]*)\}"
        matches = re.findall(pattern, llm_output)
        if len(matches) < 1:
            return "", []
        try:
            json_obj = json.loads("{" + matches[0] + "}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            return "", []
        return (
            json_obj["rephrased_question"],
            json_obj["technical_keywords"],
            json_obj["company_keywords"] if "company_keywords" in json_obj.keys() else [],
        )

    def convert_quarter_format(self, input_string):
        # Ensure both quarter formats are included eg: Q2'22 <--> Q2 2022

        # Use regular expression to match "Q<quarter>'<year>"
        match = re.match(r"Q(\d)\'(\d{2})", input_string.strip())

        if match:
            quarter = match.group(1)
            year = "20" + match.group(2)
            return f"Q{quarter} {year}"

        match = re.match(r"Q(\d) (\d{4})", input_string.strip())

        if match:
            quarter = match.group(1)
            year = match.group(2)[-2:]
            return f"Q{quarter}'{year}"

        # For case of e.g., Q1 F2023
        match = re.match(r"Q(\d) F(\d{4})", input_string.strip())

        if match:
            quarter = match.group(1)
            year = match.group(2)[-2:]
            return f"Q{quarter}'{year}"

        # For case of e.g., F2023
        match = re.match(r"F(\d{4})", input_string.strip())

        if match:
            year = match.group(1)
            return f"{year}"

        return ""

    def generate_meta_time(self, llm, prompt, query):
        """
        Generate time based keywords Metadata.
        
        Args:
            llm: Bedrock LLM to be used.
            prompt: Prompt to be used to expand query
            query: Question as by user
        Returns:
            Tuple - time_keyword_type, time kwds, explanation
        """
        prompt_format = prompt.format(
            query=query, most_recent_quarter=self.most_recent_quarter, most_recent_year=self.most_recent_year
        )
        body = json.dumps({
            "prompt": prompt_format,
            "top_k": 250,
            "top_p": 0.5,
            "max_tokens_to_sample": 512,
            "temperature": 0,
            "stop_sequences": ["Question"]
        })
        response = self.bedrock_client.invoke_model(
            modelId = 'anthropic.claude-instant-v1',
            accept = 'application/json',
            contentType = 'application/json',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        llm_output = response_body.get('completion')
        
        time_keyword_type, time_kwds, explanation = self.llm_ouput_to_json_time(llm_output)
        return time_keyword_type, time_kwds, explanation

    def llm_ouput_to_json_time(self, llm_output):
        # Use regular expressions to find content between curly braces
        pattern = r"\{([^}]*)\}"
        matches = re.findall(pattern, llm_output)
        if len(matches) < 1:
            return "", []
        try:
            json_obj = json.loads("{" + matches[0] + "}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            return "", []
        return (json_obj["time_keyword_type"], json_obj["time_keywords"], json_obj["explanation"])

    def generate_meta_kwd(self, llm, prompt, query):
        """
        Generate query based keywords Metadata.
        
        Args:
            llm: Bedrock LLM to be used.
            prompt: Prompt to be used to expand query
            query: Question as by user
        Returns:
            Tuple - query kwds
        """
        # llm_output = llm(prompt.format(query=query))
        # print(llm_output)
        # kwds = self.llm_output_kwd(llm_output)
        # return kwds
        
        # Haiku
        prompt_format = prompt.format(query=query)
        body = json.dumps({
            # "prompt": prompt_format,
            "anthropic_version": "bedrock-2023-05-31",
            "messages":[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt_format
                }]
            }],
            "top_k": 250,
            "top_p": 0.5,
            "max_tokens": 512,
            "temperature": 0,
            "stop_sequences": ["Human"]
        })
        response = self.bedrock_client.invoke_model(
            modelId = "anthropic.claude-3-haiku-20240307-v1:0",
            accept = 'application/json',
            contentType = 'application/json',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        llm_output = response_body.get('content')[0]['text'] #('content')[0]['text'] or ('completion')
        print(llm_output)
        kwds = self.llm_output_kwd(llm_output)
        return kwds

    def llm_output_kwd(self, llm_output):
        soup = BeautifulSoup(llm_output, "html.parser")
        keywords = soup.find("keywords".lower())

        keywords = re.sub("<[^<]+>", "", str(keywords))
        keywords = keywords.strip()
        if keywords:
            keywords = keywords.split(",")
            keywords = [keyword.strip() for keyword in keywords]

        return keywords

    def get_last_3_years(self, company_name, doc_type):
        assert type(company_name) == list, "company_name has to be a list"
        assert type(doc_type) == list, "doc_type has to be a list"
        metadata = pd.read_csv(configs.METADATA_PATH)
        
        # dont handle more than one company or no company
        if (len(company_name) > 1) or (len(company_name) < 1):
            return []
        company_years = []
        for doc in doc_type:
            subset_meta = metadata[
                (metadata["company_name"] == company_name[0].lower()) & (metadata["doc_type"] == doc)
            ]
            if subset_meta.shape[0] > 0:
                years = list(subset_meta["time"])
                years = [int(year) for year in years]
                years = [str(year - idx) for idx, year in enumerate([max(years)] * 3)]
                company_years.extend(years)
        return list(set(company_years))

    def doc_classification(self, time_kwds, question):
        if ("10k" in question.lower()) or ("10-k" in question.lower()):
            doc_type = "10-k"
        elif ("10q" in question.lower()) or ("10-q" in question.lower()):
            doc_type = "10-q"
        elif "earnings" in question.lower():
            doc_type = "earnings call"
        elif len(time_kwds) == 0:
            doc_type = "10-k"
        elif len(time_kwds) > 0:
            is_quarter = 0
            for kwd in time_kwds:
                if "q" in kwd.lower():
                    is_quarter += 1
                    doc_type = "10-q, earnings call"
            if not is_quarter:
                doc_type = "10-k"

        return [doc.strip() for doc in doc_type.split(",")]
