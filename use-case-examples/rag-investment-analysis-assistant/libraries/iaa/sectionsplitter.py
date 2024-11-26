from .textSplitter import TextSplitter
from .textractPdfParser import TextractPdfParser
from .text_summary import text_summary
import boto3
from io import StringIO
import csv
import logging

# Get the root logger
logger = logging.getLogger()

# Set the log level to WARNING
logger.setLevel(logging.WARNING)

class sectionSplitter():
    def __init__(self, sourcebucket, doc_path, indexId, embeddingsModel, serviceBucket):
        self.sourcebucket = sourcebucket
        self.doc_path = doc_path
        self.indexId = indexId
        self.embeddingModel = embeddingsModel
        self.serviceBucket = serviceBucket
        self.IAA_SPECIFIC = True

    def append_data_to_s3_csv(self,bucket_name, file_key, metadata):
        s3 = boto3.client('s3')
        if metadata['quarter']:
            data = [metadata['company_name'],metadata['doc_type'],metadata['time'],metadata['quarter']]
        else:
            data = [metadata['company_name'],metadata['doc_type'],metadata['time']]
        try:
            s3.head_object(Bucket=bucket_name, Key=file_key)
            exists = True
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                exists = False
            else:
                logger.error(e.response['Error'])
                raise
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        if not exists:
            writer.writerow(["company_name","doc_type", "time", "quarter"])
            writer.writerow(data)

        else:
            response = s3.get_object(Bucket=bucket_name, Key=file_key)
            lines = response['Body'].read().decode('utf-8')
            reader = csv.reader(StringIO(lines))
            for row in reader:
                writer.writerow(row)
            writer.writerow(data)

        csv_data = csv_buffer.getvalue()
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_data)

    def get_company_time_quarter_doctype(self, doc, update_metadata_to_csv=True):
        doc_metadata = {}
        company_path_data = doc.split("/")[-1].split('.')[0].split('_')
        doc_metadata["company_name"] = company_path_data[0].lower()
        doc_metadata["doc_type"] = company_path_data[2].lower()

        if 'Q' in company_path_data[1]:
            time, q = company_path_data[1].split('Q')
            doc_metadata["time"] = time
            doc_metadata["quarter"] = 'q'+q
        else:
            doc_metadata["time"] = company_path_data[1]
            doc_metadata["quarter"] = ""
        if update_metadata_to_csv:
            self.append_data_to_s3_csv(self.serviceBucket,f"metadata/{self.indexId}/company_metadata.csv",doc_metadata)
        return doc_metadata

    def doc2index(self):
        MIN_TXT_CHUNK_SIZE = 1000                           #Minimum size of chunks threshold
        SECTION_MARKER = "##"
        update_metadata_to_csv: bool = True                # Toggle True/False to write metadata to csv file in s3
        
        # call Textract PDF Parser
        textractPdfParser = TextractPdfParser(self.sourcebucket, self.doc_path, section_marker=SECTION_MARKER)    #Textract integration; parse and section Marker
        txt = textractPdfParser.get_text()

        # Text Splitter
        textSplitter = TextSplitter(MIN_TXT_CHUNK_SIZE, section_marker=SECTION_MARKER)
        text_paragraphs = textSplitter.split_txt(txt)
        file_name = self.doc_path.split("/")[-1]

        # add source document name to the Paragraph
        source_file_meta = f"""<<Paragraph>> [Source File: {file_name.replace(".pdf","")}] \n"""
        text_chunks_with_source_meta = [source_file_meta + " " + paragraph for paragraph in text_paragraphs]
        
        # Extracting Metadata from filename and write to CSV
        if self.IAA_SPECIFIC:
            doc_metadata = self.get_company_time_quarter_doctype(file_name, update_metadata_to_csv)
        
        
        final_chunks = []
        
        for p in text_chunks_with_source_meta:
            if self.IAA_SPECIFIC:
                chunk_dict = {
                "company_name": doc_metadata["company_name"], #add your own
                "doc_type":doc_metadata["doc_type"],
                "time": doc_metadata["time"],
                "paragraph": p
            }
            else:
                chunk_dict = {
                # "company_name": doc_metadata["company_name"], #add your own
                # "doc_type":doc_metadata["doc_type"],
                # "time": doc_metadata["time"],
                "paragraph": p
            }

            
            final_chunks.append(chunk_dict)
            
        return final_chunks
