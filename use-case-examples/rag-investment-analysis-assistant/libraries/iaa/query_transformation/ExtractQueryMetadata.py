import libraries.iaa.configs as configs
from libraries.iaa.query_transformation.QueryMetaExtractor import QueryMetaExtractor


class ExtractQueryMetadata:
    def extract_question_meta_v2(model_id, question, use_company_kwd=True, use_doc_type=True):
        """Extract metadata from question using QueryMetaExtractor model.

        Args:
            question (str): The input question

        Returns:
            rephrased_query (str): Rephrased version of the question
            kwds (List[str]): List of technical keywords
            time_kwds (List[str]): List of time related keywords
        """

        metaExtractor = QueryMetaExtractor(
            most_recent_quarter=configs.MOST_RECENT_QUARTER,
            most_recent_year=configs.MOST_RECENT_YEAR,
        )
        time_llm = metaExtractor.get_llm(model_id, max_tokens=1024, temperature=0)
        time_keyword_type, time_kwds, explanation = metaExtractor.generate_meta_time(
            time_llm, configs.PROMPT_QUERY_META_GENERATION_time, question)
        print(f"Time keywords generated: {time_keyword_type, time_kwds, explanation}")

        query_llm = metaExtractor.get_llm(model_id, max_tokens=1024, temperature=0.5, top_p=0.999)
        kwds = metaExtractor.generate_meta_kwd(query_llm, configs.PROMPT_TECHNICAL_KWD_v2, question)
        print(f"Query keywords generated: {kwds}")

        if time_kwds is None:
            time_kwds = ""
        
        company_meta_llm = metaExtractor.get_llm(model_id, max_tokens=1024, temperature=1)
        if use_company_kwd:
            rephrased_q, _, time_kwds, company_keywords, doc_type = metaExtractor.generate_meta(
                company_meta_llm, configs.PROMPT_QUERY_META_GENERATION_v2dot2, question, time_kwds, use_doc_type
            )
        else:
            rephrased_q, _, time_kwds, company_keywords, doc_type = metaExtractor.generate_meta(
                company_meta_llm, configs.PROMPT_QUERY_META_GENERATION_old, question, time_kwds, use_doc_type
            )

        return rephrased_q, kwds, time_kwds, company_keywords, doc_type, time_keyword_type
    
    def generate_queries(llm, query_str: str, num_queries: int = 4):
        """Generate multiple rephrasing of a user query

        Args:
            llm : rephrase llm
            query_str (str): user query
            num_queries (int, optional): number of rephrasing. Defaults to 4.

        Returns:
            str: list of generated queries
        """
        query_gen_prompt_str = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )

        query_gen_prompt = PromptTemplate(query_gen_prompt_str)
        fmt_prompt = query_gen_prompt.format(num_queries=num_queries, query=query_str)
        response = llm(fmt_prompt)
        return response
    
    def generate_hyde(llm, question):
        hyde_prompt_str = "\nHuman:\n\nPlease write a passage to answer the question\nTry to include as many key details as possible.\n\n\n {context_str} \n\n\n Assistant:"
        model_kwargs = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.3,
            "stop_sequences": ["\n\nHuman:"],
        }
        query_gen_prompt = PromptTemplate(hyde_prompt_str)
        hyde_fmt_prompt = query_gen_prompt.format(context_str=question)
        return [llm(hyde_fmt_prompt)]
