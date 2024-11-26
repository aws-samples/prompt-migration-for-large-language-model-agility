import time
from typing import Dict, List, Set
from botocore.config import Config
import os

import boto3
from textractor.entities.document import Document
import libraries.iaa.configs as configs

config = Config(
   retries = {
      'max_attempts': 10,
      'mode': 'standard'
   }
)


#Textract integration; parse and add section Marker
class TextractPdfParser:
    def __init__(
        self,
        s3_bucket_name: str,
        s3_document_key: str,
        exclude_figure_text: bool = True,
        exclude_page_header: bool = True,
        exclude_page_footer: bool = True,
        exclude_page_number: bool = True,
        skip_table: bool = False,
        save_txt_path: str = None,
        section_marker: str = "##",
        generate_markdown: bool = True,
    ):

        self.j = self._call_Textract(s3_bucket_name, s3_document_key)
        self.exclude_figure_text = exclude_figure_text
        self.exclude_page_header = exclude_page_header
        self.exclude_page_footer = exclude_page_footer
        self.exclude_page_number = exclude_page_number
        self.skip_table = skip_table
        self.save_txt_path = save_txt_path
        self.generate_markdown = generate_markdown
        self.struc_table = Document.open(self.j).tables
        self.section_marker = section_marker

        self.id_to_struc_table = {}
        tables = [x for x in self.j["Blocks"] if x["BlockType"] == "TABLE"]
        for i, table in enumerate(tables):
            self.id_to_struc_table[table["Id"]] = self.struc_table[i].to_csv()
        self.id2block = {x["Id"]: x for x in self.j["Blocks"]}
    
    
    def _call_Textract(self, s3_bucket_name, s3_document_key):
        TEXTRACT = boto3.client("textract", region_name=configs.REGION, config=config)
        response = TEXTRACT.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": s3_bucket_name,
                    "Name": s3_document_key,
                }
            },
            FeatureTypes=["TABLES", "LAYOUT"],
        )
        # Get the JobId from the response
        job_id = response["JobId"]

        # Wait for the job to complete
        print("Job still in progress...")
        def time_to_sleep():
            return 3

        while True:
            response = TEXTRACT.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]

            if status == "SUCCEEDED":
                print("Job SUCCEEDED...")
                break
            elif status == "FAILED" or status == "PARTIAL_SUCCESS":
                print("Text detection failed or partially succeeded.")
                raise Exception(f"Text detection for job {response['JobId']}failed or partially succeeded.")
            else:
                time.sleep(time_to_sleep())

        next_token = None
        if "NextToken" in response:
            next_token = response["NextToken"]
        print("Retrieving content")
        while next_token:
            response_page = TEXTRACT.get_document_analysis(JobId=job_id, NextToken=next_token)
            response["Blocks"].extend(response_page["Blocks"])
            next_token = None
            if "NextToken" in response_page:
                next_token = response_page["NextToken"]

        return response

    def _get_layout_blocks(self) -> List:
        """Get all blocks of type 'LAYOUT' and a dictionary of Ids mapped to their corresponding block."""
        layouts = [x for x in self.j["Blocks"] if x["BlockType"].startswith("LAYOUT")]
        return layouts

    def _geometry_match(self, geom1, geom2, tolerance=0.1):
        """Check if two geometries match within a given tolerance."""
        for key in ["Width", "Height", "Left", "Top"]:
            if abs(geom1[key] - geom2[key]) > tolerance:
                return False
        return True

    def _is_inside(self, inner_geom, outer_geom):
        """Check if inner geometry is fully contained within the outer geometry."""
        inner_left, inner_top, inner_right, inner_bottom = (
            inner_geom["Left"],
            inner_geom["Top"],
            inner_geom["Left"] + inner_geom["Width"],
            inner_geom["Top"] + inner_geom["Height"],
        )

        outer_left, outer_top, outer_right, outer_bottom = (
            outer_geom["Left"] - 0.1,
            outer_geom["Top"] - 0.1,
            outer_geom["Left"] + outer_geom["Width"] + 0.1,
            outer_geom["Top"] + outer_geom["Height"] + 0.1,
        )

        return (
            inner_left >= outer_left
            and inner_right <= outer_right
            and inner_top >= outer_top
            and inner_bottom <= outer_bottom
        )

    def _validate_block_skip(self, blockType: str) -> bool:
        if self.exclude_page_header and blockType == "LAYOUT_HEADER":
            return True
        elif self.exclude_page_footer and blockType == "LAYOUT_FOOTER":
            return True
        elif self.exclude_page_number and blockType == "LAYOUT_PAGE_NUMBER":
            return True
        elif self.exclude_figure_text and blockType == "LAYOUT_FIGURE":
            return True
        else:
            return False

    def get_layout_text(self, block, visited):

        texts = ""
        if block["Id"] in visited:
            return ""

        visited.add(block["Id"])
        if self._validate_block_skip(block["BlockType"]):
            return ""

        # Handle LAYOUT_TABLE type
        if not self.skip_table and block["BlockType"] == "LAYOUT_TABLE":
            # table_data = []
            # Find the matching TABLE block for the LAYOUT_TABLE
            table_block = None
            for potential_table in [b for b in self.j["Blocks"] if b["BlockType"] == "TABLE"]:
                if block["Page"] == potential_table["Page"]:
                    if self._is_inside(potential_table["Geometry"]["BoundingBox"], block["Geometry"]["BoundingBox"]):
                        table_block = potential_table
                        texts += (
                            "\n the following is a CSV table:\n{ " + self.id_to_struc_table[table_block["Id"]] + " }\n"
                        )

        elif block["BlockType"].startswith("LAYOUT") and block["BlockType"] != "LAYOUT_KEY_VALUE":
            # Get the associated LINE text for the layout
            line_texts = []
            for id_ in block["Relationships"][0]["Ids"]:
                if self.id2block[id_]["BlockType"].startswith("LAYOUT"):
                    line_texts.append(self.get_layout_text(self.id2block[id_], visited))
                    # TODO delete the layoutText
                else:
                    line_texts.append(self.id2block[id_]["Text"])

            combined_text = " ".join(line_texts)

            # Prefix with appropriate markdown
            if self.generate_markdown:
                if block["BlockType"] == "LAYOUT_TITLE":
                    combined_text = f"{self.section_marker} [source page: {block['Page']}] {combined_text}"
                elif block["BlockType"] == "LAYOUT_SECTION_HEADER":
                    combined_text = f"{self.section_marker} [source page: {block['Page']}] {combined_text}"

            texts += combined_text
        return texts

    def get_text(self) -> str:
        """Retrieve the text content in specified format. Default is CSV. Options: "csv", "markdown"."""
        # texts = []
        # page_texts = {}
        page_texts: Dict = {}
        visited: Set = set()
        layouts = self._get_layout_blocks()
        for layout in layouts:
            page_number = layout.get("Page", 1)
            if page_number not in page_texts:
                page_texts[page_number] = []

            page_texts[page_number].append(self.get_layout_text(layout, visited))

        all_txt = []
        for page_number, content in page_texts.items():
            all_txt.append("\n".join(content))
        full_txt = "\n\n".join(all_txt)
        return full_txt
