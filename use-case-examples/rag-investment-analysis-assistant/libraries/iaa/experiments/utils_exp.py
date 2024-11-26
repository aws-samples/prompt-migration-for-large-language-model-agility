import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, List

import boto3
import pandas as pd
import yaml
from langchain.llms import Bedrock
# from llama_index import PromptTemplate

import libraries.iaa.configs as configs
from libraries.iaa.answer_generation.generate_answer import GenerateAnswer

main_qa = GenerateAnswer()


def load_yaml(filename):
    # load config files
    with open(filename, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def write2text(data, file_name):
    # write chunk text data
    with open(file_name, "w") as f:
        f.write(data)


def is_valid_directory(filepath):
    p = Path(filepath)
    return p.exists() and p.is_dir()


def create_model_dir(output_dir):
    # check if dir exists, if not create
    if is_valid_directory(output_dir):
        return
    else:
        try:
            print(f"{output_dir} does not exist. Creating {output_dir}")
            os.makedirs(output_dir)
            return
        except Exception as err:
            print(err)
            print("Exiting...")
            sys.exit()


def get_titan_text_embedding(content_input):
    # convert user query into embedding
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=configs.REGION,
    )
    body = json.dumps({"inputText": content_input})
    modelId = "amazon.titan-embed-text-v1"
    accept = "application/json"
    contentType = "application/json"
    response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")
    return embedding


def rank_norm(results):
    """
    Normalize results from 2 different retrievals using keyword frequency weights
    """
    n_results = len(results)

    normalized_results = {}
    for i, doc_id in enumerate(results.keys()):
        normalized_results[doc_id] = 1 - (i / n_results)

    ranked_normalized_results = sorted(normalized_results.items(), key=lambda x: x[1], reverse=True)
    # for r in ranked_normalized_results:
    #     print(r[1])
    return dict(ranked_normalized_results)


def results_fusion(all_results, weights, top_k=15):
    """
    Fuse results from 2 different retrievals using keyword frequency weights
    """

    fusion_results = {}
    all_norm_res = []
    best_start_time = {}
    for index, results in enumerate(all_results):
        res_dict = {}
        for item in results:
            name = item["paragraph"]
            res_dict[name] = res_dict.get(name, 0) + item["score"]

        all_norm_res.append(rank_norm(res_dict))

    for index, results in enumerate(all_norm_res):
        for (_id, score) in results.items():
            fusion_results[_id] = fusion_results.get(_id, 0) + (weights[index] * score)

    # Step 3: Rank Norm
    ranked_results = sorted(fusion_results.items(), key=lambda x: x[1], reverse=True)

    res = []
    for name, score in ranked_results:
        res.append({"paragraph": name, "score": score})

    if len(res) >= top_k:
        res = res[:top_k]

    return res

def get_llm_qt(model_id, use_query_rewriting=False, use_hyde=False):
    """
    Set up LLM config based on user query rewriting choices
    """

    if use_query_rewriting:
        print("Using query rewriting\n\n")
        model_kwargs = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.3,
            "stop_sequences": ["\n\nHuman:"],
        }
    elif use_hyde:
        print("Using Hyde\n\n")
        hyde_prompt_str = "\nHuman:\n\nPlease write a passage to answer the question\nTry to include as many key details as possible.\n\n\n {context_str} \n\n\n Assistant:"
        model_kwargs = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.3,
            "stop_sequences": ["\n\nHuman:"],
        }

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=configs.REGION,
    )
    llm_qt = Bedrock(client=bedrock_runtime, model_id=model_id, model_kwargs=model_kwargs)
    return llm_qt


def get_llm_file_format(llm_model):
    if llm_model == "anthropic.claude-v2:1":
        return "v21"
    elif llm_model == "anthropic.claude-v2":
        return "v2"
    else:
        return "iv1"


def generate_file_name_experiment(kwargs, date_, ext):
    """
    Generate csv file name which encodes critical user config for experimental
    The csv file stores Question, GT answer, LLM answer, LLM citation etc
    """
    globals().update(kwargs)
    if type_qt:
        name = f"{date_}_{type_ret}_{type_qt}_{get_llm_file_format(rephrase_llm_model)}_{get_llm_file_format(generation_llm_model)}_{top_k_retrieval}_{top_k_ranking}_"
    else:
        name = f"{date_}_{type_ret}_None_{get_llm_file_format(rephrase_llm_model)}_{get_llm_file_format(generation_llm_model)}_{top_k_retrieval}_{top_k_ranking}_"

    name += f"{int(use_kw_reranker)}_{int(use_bge_reranker)}_{int(use_dynamic_ret)}_{int(use_streaming)}_{int(use_doc_type)}"

    if use_company_kwds:
        name = name + "_ckwd_"

    return name + f"{ext}.csv"


def generate_file_name_retrieval(kwargs, date_, ext):
    """
    Generate txt file name which encodes critical user config for experimental
    The txt file stores Question, GT answer, LLM answer, LLM citation and all the reranked chunks that were used for answer generation.
    """
    globals().update(kwargs)
    if type_qt:
        name = f"{date_}_{type_ret}_{type_qt}_{get_llm_file_format(rephrase_llm_model)}_{get_llm_file_format(generation_llm_model)}_{top_k_retrieval}_{top_k_ranking}_"
    else:
        name = f"{date_}_{type_ret}_None_{get_llm_file_format(rephrase_llm_model)}_{get_llm_file_format(generation_llm_model)}_{top_k_retrieval}_{top_k_ranking}_"

    name += f"{int(use_kw_reranker)}_{int(use_bge_reranker)}_{int(use_dynamic_ret)}_{int(use_streaming)}_{int(use_doc_type)}"

    if use_company_kwds:
        name = name + "_ckwd_"

    return name + f"{ext}.txt"


def add_data_point(
    csv_file,
    question,
    rephrased_question,
    q_kwds,
    time_kwds,
    company_kwds,
    gt_answer,
    answer,
    citation,
    contexts,
    end_time,
):
    """
    Add rows to the csv file for evaluation
    """
    # Check if the file exists. If not, create a new one with headers.
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=[
                "Question",
                "Rephrased Question",
                "Question Keywords",
                "Time Keywords",
                "Company Keywords",
                "Latency",
                "Bedrock Answer",
                "Answer",
                "Citation",
                "Contexts",
                "Accuracy",
                "Relevance",
                "Clarity",
            ]
        )

    # Add new question-answer pairs row by row
    new_data = {
        "Question": question,
        "Bedrock Answer": answer,
        "Rephrased Question": rephrased_question,
        "Question Keywords": q_kwds,
        "Time Keywords": time_kwds,
        "Company Keywords": company_kwds,
        "Latency": end_time,
        "Answer": gt_answer,
        "Citation": citation,
        "Contexts": contexts,
        "Accuracy": "",
        "Relevance": "",
        "Clarity": "",
    }

    # new_row = pd.DataFrame(new_data, index=[0])  # Create a new DataFrame for the row
    df.loc[len(df)] = new_data

    # df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    df.to_csv(csv_file, index=False)


def get_query_variables(use_query_rewriting, use_hyde, generation_llm_model, llm_qt=None, hyde_prompt_str=None):
    """Initialize variables required for query rephrasing

    Args:
        use_query_rewriting (boolean): boolean flag for query_rewriting
        use_hyde (boolean): boolean flag for HyDE generation
        generation_llm_model (str): Bedrock model_id of the answer generation LLM
        llm_qt (Bedrock LLM, optional): LLM from bedrock. Defaults to None.
        hyde_prompt_str (str): Prompt string for HyDE

    Returns:
        _type_: _description_
    """

    if use_query_rewriting:
        print("Using query rewriting\n\n")
        model_kwargs = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.3,
            "stop_sequences": ["\n\nHuman:"],
        }
        llm_qt = get_llm_qt(generation_llm_model, model_kwargs)
    elif use_hyde:
        print("Using Hyde\n\n")
        hyde_prompt_str = "\nHuman:\n\nPlease write a passage to answer the question\nTry to include as many key details as possible.\n\n\n {context_str} \n\n\n Assistant:"
        model_kwargs = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.3,
            "stop_sequences": ["\n\nHuman:"],
        }
        llm_qt = get_llm_qt(generation_llm_model, model_kwargs)

    return llm_qt, hyde_prompt_str


def update_config(exp_config: dict, type_ret: str, top_k_ranking: str, use_doc_type: bool, emb_name="TT_chunk_emb"):
    """
    Update user config with composite variables for downstream tasks
    """
    exp_config["use_text"] = True if type_ret == "text" else False
    exp_config["use_embedding"] = True if type_ret == "embb" else False
    exp_config["use_query_rewriting"] = True if exp_config["type_qt"] == "qw" else False
    exp_config["use_hyde"] = True if exp_config["type_qt"] == "hyde" else False
    exp_config["type_ret"], exp_config["top_k_ranking"], exp_config["use_doc_type"] = (
        type_ret,
        top_k_ranking,
        use_doc_type,
    )
    exp_config["use_rephrased_query"] = True if exp_config["type_qt"] is None else False
    exp_config["llm_qt"], exp_config["hyde_prompt_str"] = get_query_variables(
        exp_config["use_query_rewriting"], exp_config["use_hyde"], exp_config["generation_llm_model"]
    )
    exp_config["i"], exp_config["add_end"] = 0, 0
    exp_config["emb_name"] = emb_name if "emb_name" not in exp_config else exp_config["emb_name"]
    return exp_config


def parse_time_kwds(time_kwds) -> list:
    """Given LLM generated time keywords, extract quarter and year

    Args:
        time_kwds (list): LLM generated time keywords

    Returns:
        list: List of tuples with year and quarter
    """
    set_tuple = set([])
    for kwd in time_kwds:
        match = re.match(r"Q(\d)\'(\d{2})", kwd.strip())
        if match:
            quarter = "q" + match.group(1)
            year = "20" + match.group(2)
            set_tuple.add((year, quarter))
            continue

        match = re.match(r"Q(\d) (\d{4})", kwd.strip())
        if match:
            quarter = "q" + match.group(1)
            year = match.group(2)
            set_tuple.add((year, quarter))
            continue

        match = re.match(r"Q(\d) F(\d{4})", kwd.strip())
        if match:
            quarter = "q" + match.group(1)
            year = match.group(2)
            set_tuple.add((year, quarter))
            continue

        match = re.search(r"(20\d{2})", kwd.strip())
        if match:
            quarter = ""
            year = match.group(1)
            set_tuple.add((year, quarter))
            continue

    return list(set_tuple)


def load_exp_config(query_file="../path/to/grount_truth.csv", exp_config_file="../iaa/exp_config/test.yml"):
    """Load config file for an experiment

    Args:
        query_file (str, optional): csv file with iaa's QnA. 
        exp_config_file (str, optional): File with specific exp config. Defaults to "../iaa/exp_config/test.yml".

    Returns:
        tuple: return GT dataframe and config dictionary
    """
    # gt = pd.read_csv(query_file)
    # gt.rename(columns={"Prompt": "question", "Reference Answer": "answer"}, inplace=True)
    exp_config = load_yaml(exp_config_file)

    return exp_config
