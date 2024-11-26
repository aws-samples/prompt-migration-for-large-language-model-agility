import json
import re
import string
import time
from dataclasses import dataclass, field
from queue import Queue
# from chatResponder import ChatResponder

import boto3
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
# from langchain.chains.summarize import load_summarize_chain
# from langchain.docstore.document import Document
# from llama_index import PromptTemplate

import libraries.iaa.configs as configs
from libraries.iaa.experiments.utils_exp import (
    add_data_point,
    get_llm_qt,
    get_titan_text_embedding,
    main_qa,
    parse_time_kwds,
    results_fusion,
    write2text,
)
from libraries.iaa.query_transformation.ExtractQueryMetadata import ExtractQueryMetadata
from libraries.iaa.reranker.SearchRanker import SearchRanker
from libraries.iaa.retrieval.OpenSearchRetrieval import OpenSearchRetrieval


@dataclass
class RagQA:
    """
    Class that facilitates answer generation by question. Stores all intermediate metadata associated with a question
    """
    question: str
    # gt_answer: str
    # company: str
    exp_config: dict
    # out_csv_file: str
    # chunk_result_folder: str
    # queue: Queue
    # chat_responder: ChatResponder
    data: str = ""
    kwds_list: list = field(default_factory=list)
    time_kwds_list: list = field(default_factory=list)
    company_kwds_list: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    top_k_contexts: list = field(default_factory=list)
    end_time: int = 0
    time_count: int = 0
    write_data: bool = True
    

    def rephrase_query(self):
        """Rephrase a question based on user's query transform preference
        list: list of rephrased questions
        """
        for key in self.exp_config:
            setattr(self, key, self.exp_config[key])
        self.t0 = time.time()
        if self.use_query_rewriting:
            queries = ExtractQueryMetadata.generate_queries(self.llm_qt, self.question, num_queries=3)
            q_list = queries.split("\n\n")[1:]
            print(f"qw {len(queries.split(' '))}")
        elif self.use_hyde:
            q_list = ExtractQueryMetadata.generate_hyde(self.llm_qt, self.question)
            print(f"hyde {len(q_list[0].split(' '))}")
        else:
            q_list = [self.question]

        return q_list

    def get_list_variables(self, company_kwds, time_kwds, kwds):
        """Update query metadata

        Args:
            company_kwds (list): company names associated with a question
            time_kwds (list): time keywords associated with a question
            kwds (list): technical keywords associated with a question

        Returns:
            _type_: _description_
        """
        # if company_kwds:
        #     self.company_kwds_list.extend([f"Source File: {x}" for x in company_kwds])
        if company_kwds:
            self.company_kwds_list.extend(set(company_kwds))
        self.kwds_list.extend(set(kwds))
        self.time_kwds_list.extend(set(time_kwds))
        time_kwds_tuples = parse_time_kwds(self.time_kwds_list)
        time_key_in_years = list(set([x[0] for x in time_kwds_tuples]))
        self.time_count = len(time_key_in_years)
        q_kwds = []

        return time_key_in_years, q_kwds

    def get_contexts(self, q):
        """Given a question first generate relevant metadata and then
        retrieve all relevant chunks from search index

        Args:
            q (str): user query

        Returns:
            tuple: (updated time_kwds, rephrased query)
            Context is updated as class field
        """
        try:
            t0 = time.time()
            company_kwds_list_prev, kwds_list_prev, time_kwds_list_prev, context_prev = (
                self.company_kwds_list,
                self.kwds_list,
                self.time_kwds_list,
                self.contexts,
            )
            print(f"self.rephrase_llm_model {self.rephrase_llm_model}")
            (
                rephrased_query,
                kwds,
                time_kwds,
                company_kwds,
                self.doc_type,
                time_keyword_type,
            ) = ExtractQueryMetadata.extract_question_meta_v2(
                self.rephrase_llm_model, q, self.use_company_kwds, self.use_doc_type
            )
            
            time_key_in_years, q_kwds = self.get_list_variables(company_kwds, time_kwds, kwds)
            end_time = time.time()
            print(f"**** Time taken for query trans {end_time - t0}")
            
            t0 = time.time()
            self.contexts = self.get_context(q, q_kwds, time_keyword_type, time_key_in_years, kwds, time_kwds, company_kwds)
            end_time = time.time()
            print(f"**** Time taken for retrieval {end_time - t0}")
            
            if len(self.contexts) == 0:
                print("Retrying as no context was retrieved")
                self.company_kwds_list, self.kwds_list, self.time_kwds_list, self.contexts = (
                    company_kwds_list_prev,
                    kwds_list_prev,
                    time_kwds_list_prev,
                    context_prev,
                )
                return self.get_contexts(q)

            return (time_kwds, rephrased_query)
        except Exception as err:
            raise err

    def get_context(self, q, q_kwds, time_keyword_type, time_key_in_years, kwds, time_kwds, company_kwds):
        """Given a question retrieve all relevant chunks from search index"""
        if self.use_text:
            retriever = OpenSearchRetrieval(configs.OPEN_SEARCH_HOST, self.index_name_text)
            self.contexts.extend(
                [
                    context["paragraph"]
                    for context in retriever.retrieve_text(
                        kwds,
                        time_kwds,
                        company_kwds,
                        self.doc_type,
                        self.top_k_retrieval,
                        self.use_company_kwds,
                        self.use_doc_type,
                    )
                ]
            )
        elif self.use_embedding:
            # Set the top k retrieval based on time keywords
            if self.use_dynamic_ret:
                self.top_k_retrieval = 10 * self.time_count if self.time_count > 0 else 30
                print(f"top_k_retrieval {self.top_k_retrieval}")

            retriever = OpenSearchRetrieval(configs.OPEN_SEARCH_HOST, self.index_name_embb)
            embedding = get_titan_text_embedding(q)
            semantic_time_kwds = time_key_in_years
            if time_keyword_type == "none":
                semantic_time_kwds = []
                q_kwds = []
            self.contexts.extend(
                [
                    context["paragraph"]
                    for context in retriever.retrieve_semantic(
                        self.emb_name,
                        embedding,
                        time_key_in_years,
                        company_kwds,
                        self.doc_type,
                        q_kwds,
                        top_k=self.top_k_retrieval,
                        use_company_kwds=self.use_company_kwds,
                        use_doc_type=self.use_doc_type,
                    )
                ]
            )
        else:
            retriever_embb = OpenSearchRetrieval(configs.OPEN_SEARCH_HOST, self.index_name_embb)
            embedding = get_titan_text_embedding(q)
            contexts_sem = retriever_embb.retrieve_semantic(
                self.emb_name,
                embedding,
                time_kwds,
                company_kwds,
                self.doc_type,
                q_kwds,
                top_k=self.top_k_retrieval,
                use_company_kwds=self.use_company_kwds,
                use_doc_type=self.use_doc_type,
            )

            retriever_text = OpenSearchRetrieval(configs.OPEN_SEARCH_HOST, self.index_name_embb)
            contexts_text = retriever_text.retrieve_text(
                kwds,
                time_kwds,
                company_kwds,
                self.doc_type,
                self.top_k_retrieval,
                self.use_company_kwds,
                self.use_doc_type,
            )
            self.contexts.extend(
                [
                    context["paragraph"]
                    for context in results_fusion([contexts_sem, contexts_text], [0.7, 0.3], top_k=self.top_k_retrieval)
                ]
            )

        return self.contexts

    def rerank_context(self, rephrased_query, endpoint_name="bge_enndpoint_name_if_using_bge"):
        # Set the top k ranking based on time keywords
        t0 = time.time()
        if self.use_dynamic_ret:
            self.top_k_ranking = self.time_count * 5 if self.time_count > 0 else 30
            print(f"top_k_ranking {self.top_k_ranking}")

        if self.use_kw_reranker:
            print("With keyword reranker")
            # Rank the contexts
            ranker = SearchRanker()
            combined_kwds = []
            combined_kwds.extend(set(self.kwds_list))
            combined_kwds.extend(set(self.time_kwds_list))
            combined_kwds.extend(set(self.company_kwds_list))
            ranked_contexts = ranker.rank_by_word_frequency(combined_kwds, self.contexts)
        elif self.use_bge_reranker:
            print("With custom BGE reranker")
            sm_client = boto3.client("sagemaker-runtime")

            payload = [(rephrased_query, context) for context in self.contexts]
            payload = {"pairs": payload}
            payload = json.dumps(payload)
            response = sm_client.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="application/json", Body=payload
            )
            result = json.loads(response["Body"].read())["scores"]
            idx_sort = list(np.argsort(result)[::-1])
            ranked_contexts = [self.contexts[idx] for idx in idx_sort]
        else:
            ranked_contexts = self.contexts

        # ***** Reduce chunks using LangChain map_reduce ****
        if self.use_map_reduce:
            # To update map and reduce prompts uncomment and use this method
            # new_context = map_reduce_chain(ranked_contexts[:20])

            # Use inbuilt map reduce prompts
            model_kwargs = {
                "max_tokens_to_sample": 4096,
                "temperature": 0,
                "top_k": 50,
                "stop_sequences": ["\n\nHuman:"],
            }
            llm_mp = get_llm_qt("anthropic.claude-instant-v1", model_kwargs)

            chain = load_summarize_chain(llm_mp, chain_type="map_reduce")
            doc_contexts = [Document(page_content=t) for t in ranked_contexts[: self.top_k_ranking]]
            new_context = chain.batch([doc_contexts])

        # Get the topk contexts
        if self.use_map_reduce:
            top_k_contexts = new_context
        else:
            top_k_contexts = ranked_contexts[: self.top_k_ranking]
        
        end_time = time.time()
        print(f"**** Time taken for reranking {end_time - t0}")
        return ranked_contexts, top_k_contexts

    def get_llm_answer(self, q_list, time_kwds, top_k_contexts, rephrased_query):
        t0 = time.time()
        if self.use_rephrased_query:
            prompt = main_qa.format_prompt_v2(self.question, time_kwds, top_k_contexts, rephrased_query)
        else:
            prompt = main_qa.format_prompt_v2(self.question, time_kwds, top_k_contexts, q_list)
        
        unaccetable_text = ["<answer", ">", "<pages", ">", "<src", "<src>", "<answer>", "<pages>",
                                "</src>", "</answer>", "</pages>", "</answer", "</pages", "</src", "</",
                                "answer", "src", "pages"]

        try:
            if self.use_streaming:
                stream = main_qa.get_llm_answer_stream(self.generation_llm_model, 4096, 0.1, prompt).get("body")
                output = []
                la = True
                if stream:
                    for event in stream:
                        chunk = event.get("chunk")
                        if chunk:
                            if la:
                                self.end_time = time.time() - self.t0
                                print(f"\n**** Stream Start {self.end_time} ****\n")
                                la = False
                            chunk_obj = json.loads(chunk.get("bytes").decode())
                            if self.generation_llm_model in ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"]:
                                try:
                                    if len(chunk_obj['delta']['text']) > 0:
                                        text = chunk_obj['delta']['text']
                                        output.append(text)
                                except:
                                    continue
                            else:
                                text = chunk_obj["completion"]
                                output.append(text)
                            if text in unaccetable_text:
                                continue
                            if not configs.LOCAL_MODE:
                                self.chat_responder.publish_agent_partial_message(text.replace("</", ""))
                prediction = "".join(output)
            else:
                print(f"self.generation_llm_model {self.generation_llm_model}")
                # For bedrock invoke Haiku/instant-v1
                prediction = main_qa.get_llm_answer(model_id=self.generation_llm_model, max_tokens=4096, temperature=0.1, prompt=prompt)
            end_time = time.time()
            print(prediction)
            print(f"**** Time taken for ans generation {end_time - t0}")
            if not configs.LOCAL_MODE:
                answer, source = self.parse_generation(prediction)
                bucket_name = str(configs.BUCKET_NAME)
                file_name = str(source)
                if "Source File" in file_name:
                    file_name = file_name.split(":")[1].strip("[] ")
                file_names_str = file_name
                file_locations = self.find_files_in_bucket(bucket_name, file_names_str)
                if file_locations:
                    for file_location in file_locations:
                        print(f'File found at: {file_location}')
                        presigned_url = self.create_presigned_url(bucket_name, file_location)
                        if presigned_url:
                            print(f'Pre-signed URL: {presigned_url}')
                            sourcefile = file_location.split("/")[-1]
                            print(f'sourcefile: {sourcefile}')
                            self.chat_responder.publish_agent_send_sources({"source": sourcefile, "sourceurl": presigned_url})
                else:
                    print('No files found in the bucket.')


                self.chat_responder.publish_agent_message(prediction)
                self.chat_responder.publish_agent_stop_responding()
            return prediction
        except Exception as err:
            raise err
        finally:
            if not configs.LOCAL_MODE:
                self.chat_responder.publish_agent_stop_responding()

    def parse_generation(self, llm_prediction):
        soup = BeautifulSoup(llm_prediction, "html.parser")
        answer = soup.find("answer".lower())
        page_source = soup.find("pages".lower())
        source = soup.find("src".lower())
        
        answer = re.sub("<[^<]+>", "", str(answer))
        page_source = re.sub("<[^<]+>", "", str(page_source))
        source = re.sub("<[^<]+>", "", str(source))

        return answer+'\n'+page_source, source

    def generate_answer(self):
        try:
            debug_time = {}
            t0_transform = time.time()
            q_list = self.rephrase_query()
            for q in q_list:

                time_kwds, rephrased_query = self.get_contexts(q)

            t_end_time_transform = time.time()
            t0_rerank = time.time()
            ranked_contexts, top_k_contexts = self.rerank_context(rephrased_query, configs.ENDPOINT_NAME)
            t_end_time_rerank = time.time()
            t0_ansgen = time.time()
            prediction = self.get_llm_answer(q_list, time_kwds, top_k_contexts, rephrased_query)
            # t_end_time_retrieval = time.time()
            t_end_time_ansgen = time.time()
            # print(prediction)
            debug_time['queryTransformationRetrieval'] = str(t_end_time_transform - t0_transform)
            debug_time['reranking'] = str(t_end_time_rerank - t0_rerank)
            debug_time['answerGeneration'] = str(t_end_time_ansgen - t0_ansgen)
            answer, source = self.parse_generation(prediction)
            self.write_data = False
            if self.write_data:
                # TODO: Modify to read a file from s3 and add datapoint to that file
                add_data_point(
                    self.out_csv_file,
                    self.question,
                    q_list,
                    self.kwds_list,
                    self.time_kwds_list,
                    self.company_kwds_list,
                    self.gt_answer,
                    answer,
                    "",
                    top_k_contexts,
                    self.end_time,
                )
                self.data += f"Question: {rephrased_query}" + "\n" + "\n"
                self.data += "---------------------------" + "\n"
                self.data += f"GT Answer: {self.gt_answer}" + "\n" + "\n"
                self.data += "---------------------------" + "\n"
                self.data += f"LLM Prediction: {prediction}" + "\n" + "\n"
                self.data += "---------------------------" + "\n"
                self.data += f"Retrieved Chunks: " + "\n"
                for idx, context in enumerate(ranked_contexts):
                    self.data += f"Context {idx}: " + context
                    self.data += "---------------------------" + "\n"

                question_formatted = self.question.translate(str.maketrans("", "", string.punctuation))
                write2text(self.data, self.chunk_result_folder + f"{question_formatted.replace(' ','_')}.txt")

            return answer, source, debug_time
        except Exception as err:
            raise err
            
    def find_files_in_bucket(self, bucket_name, file_names_str):
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        operation_parameters = {'Bucket': bucket_name}
        found_files = []
        file_names = file_names_str.split(',')  # Split the string into a list of file names
        for page in paginator.paginate(**operation_parameters):
            if 'Contents' in page:
                for obj in page['Contents']:
                    for file_name in file_names:
                        if file_name.strip() in obj['Key']:  # Strip any leading/trailing whitespace
                            found_files.append(obj['Key'])
        return found_files

    def create_presigned_url(self, bucket_name, object_key, expiration=3600):
        s3 = boto3.client('s3')
        try:
            response = s3.generate_presigned_url('get_object',
                                                Params={'Bucket': bucket_name,
                                                        'Key': object_key},
                                                ExpiresIn=expiration)
        except Exception as e:
            print(f'Error generating pre-signed URL: {e}')
            return None
        return response