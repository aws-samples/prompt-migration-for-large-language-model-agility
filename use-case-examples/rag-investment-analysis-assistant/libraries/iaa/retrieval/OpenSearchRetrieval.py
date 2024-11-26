import re
from typing import Any, List
import os
import libraries.iaa.configs as configs
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth

region=configs.REGION #update if region is not us-east-1
class OpenSearchRetrieval:
    """
    This class helps with retrieving chunks from Opensearch Service using text based
    or semantic based search bodies
    
    Args:
        host: OpenSearch host
        index_name: Name of the index to retrieve from in Opensearch Service
    """
    def __init__(self, host, index_name):
        self.host = host
        self.index_name = index_name
        self.client = self.auth_opensearch(host)
        self.verify_index()

    def verify_index(self):
        if not self.client.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' doesn't exists.")

    def auth_opensearch(self, host, service="es", region=region) -> OpenSearch:
        """
        Authenticate with OpenSearch.
        Args:
            host (str): OpenSearch host
            service (str): OpenSearch service
            region (str): OpenSearch region

        Returns:
            OpenSearch: OpenSearch client
        """

        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, region, "aoss")
        # awsauth = AWS4Auth(
        #     credentials.access_key,
        #     credentials.secret_key,
        #     region,
        #     service,
        #     session_token=credentials.token,
        # )
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=3000,
        )
        return client

    def parse_time_kwds(self, time_kwds: List[Any]) -> list:
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

    def create_search_body_text(
        self,
        kwds: List[Any],
        time_kwds: List[Any],
        company_kwds: List[Any],
        doc_type: List[Any],
        k: int,
        use_company_kwds=True,
        use_doc_type=False,
    ) -> dict:
        """
        Create OpenSearch body for OpenSearch call

        Args:
            kwds (str): text to search
            time_kwds: time keyowrds to search against time field
            company_kwds: company name to search 
            doc_type: document type to search
            k (int): number of results to return
            use_company_kwds: Boolean to use company field
            use_doc_type: Boolean to use doc type field
        Returns:
            Dict: OpenSearch body
        """
        if use_company_kwds:
            time_kwds_tuples = self.parse_time_kwds(time_kwds)
            time_kwds_years = list(set([x[0] for x in time_kwds_tuples]))

            print(f"Using exact matching in text search with {time_kwds_years}, {company_kwds}")
            text_search_params = {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [{"match": {"text": x}} for x in kwds],
                                "minimum_should_match": "70%",
                            }
                        },
                        {
                            "bool": {
                                "should": [{"match": {"time": x}} for x in time_kwds_years],
                            }
                        },
                        {
                            "bool": {
                                "should": [{"match": {"company_name": x}} for x in company_kwds],
                            }
                        },
                    ]
                }
            }
            if use_doc_type:
                print("Using doc_type in search")
                text_search_params["bool"]["must"].append(
                    {
                        "bool": {
                            "should": [{"match": {"doc_type": {"query": x, "operator": "and"}}} for x in doc_type],
                        }
                    }
                )

        else:
            print("Using non-exact matching in text search")
            text_search_params = {
                "bool": {
                    "should": [
                        {
                            "bool": {
                                "should": [{"match": {"text": x}} for x in kwds],
                                "minimum_should_match": "50%",
                            }
                        },
                        {
                            "bool": {
                                "should": [{"match": {"text": x}} for x in time_kwds],
                                "minimum_should_match": "25%",
                            }
                        },
                    ]
                }
            }
            if use_doc_type:
                print("Using doc_type in search")
                text_search_params["bool"]["must"].append(
                    {
                        "bool": {
                            "should": [{"match": {"doc_type": {"query": x, "operator": "and"}}} for x in doc_type],
                        }
                    }
                )

        body = {"size": k, "query": text_search_params}

        return body

    def create_search_body_semantic(
        self,
        vector_field: str,
        embedding: List[float],
        k: int,
        company_kwds,
        time_kwds,
        doc_type=[],
        q_kwds=[],
        use_company_kwds=True,
        use_doc_type=False,
    ) -> dict:
        """
        Create OpenSearch body for semantic retrieval

        Args:
            vector_field (str): field to search
            embedding (List[float]): query embedding
            k (int): number of results to return
            company_kwds: company name to search 
            time_kwds: time keyowrds to search against time field
            doc_type: document type to search
            q_kwds: quarter to search
            use_company_kwds: Boolean to use company field
            use_doc_type: Boolean to use doc type field
        Returns:
            Dict: OpenSearch body
        """
        if use_company_kwds:
            print(f"Using exact matching in semantic search with {time_kwds}, {company_kwds}")
            semantic_query_params = {
                "bool": {
                    "should": [
                        {"knn": {vector_field: {"vector": embedding, "k": len(embedding)}}},
                        {
                            "bool": {
                                "must": [{"match": {"time": x}} for x in time_kwds],
                            }
                        },
                        {
                            "bool": {
                                "should": [{"match": {"quarter": x}} for x in q_kwds],
                            }
                        },
                        {
                            "bool": {
                                "should": [{"match": {"company_name": x}} for x in company_kwds],
                            }
                        },
                    ]
                }
            }
            if use_doc_type:
                print(f"Using doc_type exact matching in semantic search with {doc_type}")
                semantic_query_params["bool"]["should"].append(
                    {
                        "bool": {
                            "should": [{"match": {"doc_type": {"query": x, "operator": "and"}}} for x in doc_type],
                        }
                    }
                )

            search_body = {
                "size": k,
                "query": semantic_query_params,
            }
        else:
            if use_doc_type:
                print(f"Using doc_type exact matching in semantic search with {doc_type}")
                semantic_query_params = {
                    "bool": {
                        "should": [
                            {"knn": {vector_field: {"vector": embedding, "k": len(embedding)}}},
                            {
                                "bool": {
                                    "should": [{"match": {"time": x}} for x in time_kwds],
                                }
                            },
                            {
                                "bool": {
                                    "should": [
                                        {"match": {"doc_type": {"query": x, "operator": "and"}}} for x in doc_type
                                    ],
                                }
                            },
                        ]
                    }
                }
                search_body = {
                    "size": k,
                    "query": semantic_query_params,
                }
            else:
                print("Use no doc_type and no company time kwd")
                semantic_query_params = {
                    "bool": {
                        "should": [
                            {"knn": {vector_field: {"vector": embedding, "k": len(embedding)}}},
                            {
                                "bool": {
                                    "should": [{"match": {"time": x}} for x in time_kwds],
                                }
                            },
                        ]
                    }
                }
                search_body = {
                    "size": k,
                    "query": semantic_query_params,
                }
        return search_body

    def retrieve_text(
        self,
        kwds: list,
        time_kwds: list,
        company_kwds: list,
        doc_type: list,
        top_k: int,
        use_company_kwds: bool = True,
        use_doc_type: bool = True,
    ):
        """
        Get context/chunks from OpenSearch

        Args:
            kwds (str): text to search
            time_kwds: time keyowrds to search against time field
            company_kwds: company name to search 
            doc_type: document type to search
            top_k (int): number of results to return
            use_company_kwds: Boolean to use company field
            use_doc_type: Boolean to use doc type field
        Returns:
            str: context
        """
        # create OpenSearch request body
        body = self.create_search_body_text(
            kwds, time_kwds, company_kwds, doc_type, top_k, use_company_kwds, use_doc_type
        )

        # call opensearch API
        os_res = self.client.search(request_timeout=30, index=self.index_name, body=body)

        contexts = []
        hits = os_res["hits"]["hits"]
        for hit in hits:
            contexts.append(
                {
                    "paragraph": hit["_source"]["text"],
                    "score": hit["_score"],
                }
            )

        return contexts

    def retrieve_semantic(
        self,
        vector_field,
        embedding,
        time_kwds,
        company_kwds,
        doc_type,
        q_kwds,
        top_k: int,
        use_company_kwds: bool = True,
        use_doc_type: bool = False,
    ):
        """
        Get context from OpenSearch

        Args:
            vector_field (str): field to search
            embedding (List[float]): query embedding
            time_kwds: time keyowrds to search against time field
            company_kwds: company name to search 
            doc_type: document type to search
            q_kwds: quarter to search
            top_k (int): number of results to return
            use_company_kwds: Boolean to use company field
            use_doc_type: Boolean to use doc type field
        Returns:
            str: context
        """
        # create OpenSearch request body
        body = self.create_search_body_semantic(
            vector_field, embedding, top_k, company_kwds, time_kwds, doc_type, q_kwds, use_company_kwds, use_doc_type
        )
        
        # call opensearch API
        os_res = self.client.search(request_timeout=30, index=self.index_name, body=body)
        contexts = []
        hits = os_res["hits"]["hits"]
        for hit in hits:
            contexts.append(
                {
                    "paragraph": hit["_source"]["text"],
                    "score": hit["_score"],
                }
            )
        return contexts
